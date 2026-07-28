"""
algorithms/ppo/env.py
محیط Gymnasium طبق بخش ۱۱ سند. هر گام env = یک DECISION_TICK از موتور
(بخش ۱۱.۱: عامل هر DECISION_INTERVAL_SEC سه نوع تصمیم می‌گیرد).

Action (بخش ۱۱.۳): MultiDiscrete
    ۱۵ بعد اول: برای هر سرویس {0=NO_CHANGE, 1=SCALE_UP, 2=SCALE_DOWN}
    ۱۰ بعد بعدی: برای هر سرور {0=NO_CHANGE, 1=TURN_ON, 2=TURN_OFF}

Reward (بخش ۱۱.۴): ترکیب وزن‌دار منفی از ۴ معیار + جریمه‌ی درخواست ردشده +
    جریمه‌ی ثابت کوچک هر اکشن (برای جلوگیری از flapping بی‌مورد عامل).

*** طبق تأیید معماری، از sb3-contrib's MaskablePPO استفاده می‌شود که قبل از
sample کردن اکشن، گزینه‌های نامعتبر را از توزیع احتمال حذف می‌کند؛ به همین
دلیل action_masks() پیاده‌سازی شده (اینترفیس مورد انتظار MaskableMultiDiscrete
در sb3-contrib).
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common.config import CFG
from common.models import ServerState, ReplicaState
from common.state_builder import build_state_vector, STATE_DIM
from algorithms.base import ScaleAction, ProvisionAction, ProvisionActionType
from simulator.engine import SimulationEngine

N_SERVICES = CFG.n_services
N_SERVERS = CFG.n_servers
_SERVICE_IDS = sorted(CFG.services_info.keys())
_SERVER_IDS = sorted(CFG.server_info.keys())

# نگاشت اکشن گسسته -> enum
_SCALE_MAP = {0: ScaleAction.NO_CHANGE, 1: ScaleAction.SCALE_UP, 2: ScaleAction.SCALE_DOWN}
_PROVISION_MAP = {0: ProvisionActionType.NO_CHANGE, 1: ProvisionActionType.TURN_ON,
                   2: ProvisionActionType.TURN_OFF}


class EdgeResourceEnv(gym.Env):
    """
    events_df_provider: callable بدون آرگومان که یک DataFrame رویداد تازه
    برمی‌گرداند (برای آموزش: هر اپیزود می‌تواند یک بازه‌ی تصادفی از تایم‌لاین
    سه‌روزه‌ی train باشد - نگاه کنید به train.py).
    """
    metadata = {"render_modes": []}

    def __init__(self, events_df_provider, teacher_algorithm=None):
        super().__init__()
        self.events_df_provider = events_df_provider
        # از initial_placement/select_replica مشترک AlgorithmBase استفاده می‌شود؛
        # چون این دو متد abstract نیستند نیازی به نمونه‌ی کامل الگوریتم نداریم،
        # ولی چون AlgorithmBase انتزاعی است یک پیاده‌ساز حداقلی لازم است:
        self._shared_algo = teacher_algorithm or _MinimalSharedAlgorithm()

        self.action_space = spaces.MultiDiscrete([3] * N_SERVICES + [3] * N_SERVERS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)

        self.engine: SimulationEngine | None = None
        self._last_snapshot = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        events_df = self.events_df_provider()
        self.engine = SimulationEngine(events_df, self._shared_algo, "ppo_train")
        self.engine.prime()
        snapshot = self.engine.peek_snapshot()  # observation اولیه، بدون اجرای واقعی تیک
        self._last_snapshot = snapshot
        obs = build_state_vector(snapshot, self.engine.servers)
        return obs, {}

    def step(self, action):
        service_actions = {sid: _SCALE_MAP[int(action[i])] for i, sid in enumerate(_SERVICE_IDS)}
        server_actions = {sid: _PROVISION_MAP[int(action[N_SERVICES + j])]
                           for j, sid in enumerate(_SERVER_IDS)}

        n_actions_taken = sum(1 for a in service_actions.values() if a != ScaleAction.NO_CHANGE)
        provision_action = ProvisionAction(ProvisionActionType.NO_CHANGE)
        for sid, ptype in server_actions.items():
            if ptype != ProvisionActionType.NO_CHANGE:
                provision_action = ProvisionAction(ptype, sid)
                n_actions_taken += 1
                break  # طبق بخش ۱۱.۳ در هر تیک حداکثر یک اکشن provisioning اعمال می‌شود

        external = {"provision": provision_action, "scale": service_actions}
        snapshot, done = self.engine.step(external_actions=external)

        if done:
            obs = np.zeros(STATE_DIM, dtype=np.float32)
            reward = 0.0
            terminated = True
        else:
            obs = build_state_vector(snapshot, self.engine.servers)
            reward = self._compute_reward(snapshot, n_actions_taken)
            terminated = False

        self._last_snapshot = snapshot
        return obs, reward, terminated, False, {}

    # ------------------------------------------------------------------
    def _compute_reward(self, snapshot: dict, n_actions_taken: int) -> float:
        """بخش ۱۱.۴."""
        w = CFG.ppo_reward_weights
        g = snapshot["global"]

        avg_dv_rate = (sum(s["deadline_violation_rate"] for s in snapshot["services"].values())
                       / max(len(snapshot["services"]), 1))
        active_utils = [s["utilization"] for s in snapshot["servers"].values()
                         if s["state"] == ServerState.ACTIVE]
        load_cv = 0.0
        if len(active_utils) >= 2 and np.mean(active_utils) > 0:
            load_cv = float(np.std(active_utils) / np.mean(active_utils))

        norm_rt = min(g["avg_response_time_recent"] / 300.0, 2.0)
        norm_energy = min(g["energy_recent_joule"] / 12_000.0, 2.0)  # *** کالیبره‌شده - نگاه کنید common/state_builder.py
        norm_lb = min(load_cv, 2.0)

        penalty = (w["w1_response_time"] * norm_rt +
                   w["w2_deadline"] * avg_dv_rate +
                   w["w3_energy"] * norm_energy +
                   w["w4_load_balance"] * norm_lb)
        penalty += CFG.ppo_penalty_per_rejected * g["num_rejected_recent"]
        penalty += CFG.ppo_penalty_per_action * n_actions_taken
        return -float(penalty)

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """اینترفیس مورد انتظار sb3_contrib.common.wrappers.ActionMasker /
        MaskableMultiDiscrete: یک آرایه‌ی بولی مسطح به طول sum(nvec)."""
        masks = []
        snapshot = self._last_snapshot
        for sid in _SERVICE_IDS:
            sv = snapshot["services"][sid]
            can_up = self._any_server_can_host(sid)
            can_down = sv["n_replicas"] > 1
            masks.extend([True, can_up, can_down])  # NO_CHANGE همیشه مجاز است
        for sid in _SERVER_IDS:
            st = snapshot["servers"][sid]["state"]
            can_on = st == ServerState.OFF
            can_off = st == ServerState.ACTIVE
            masks.extend([True, can_on, can_off])
        return np.array(masks, dtype=bool)

    def _any_server_can_host(self, service_id: int) -> bool:
        cpu = CFG.services_info[service_id]["cpu_demand"]
        return any(s.can_host(service_id, cpu) for s in self.engine.servers.values())


class _MinimalSharedAlgorithm:
    """
    پیاده‌سازی حداقلی فقط برای initial_placement/select_replica (که طبق سند
    بین همه‌ی الگوریتم‌ها مشترک است - AlgorithmBase پیش‌فرض دارد). متدهای
    scale_decision/provision_decision/... اینجا صدا زده نمی‌شوند چون
    EdgeResourceEnv مستقیماً external_actions به engine.step() می‌دهد.
    """
    def __init__(self):
        from algorithms.base import AlgorithmBase

        class _Impl(AlgorithmBase):
            name = "ppo_shared"

            def scale_decision(self, *a, **k):
                raise NotImplementedError

            def provision_decision(self, *a, **k):
                raise NotImplementedError

            def select_placement_server(self, service_id, servers):
                """بخش ۱۱.۳: «انتخاب سرور مقصد با بیشترین ظرفیت آزاد» - قانون مشترک."""
                from common.models import ServerState
                from common.config import CFG as _CFG
                cpu = _CFG.services_info[service_id]["cpu_demand"]
                candidates = [s for s in servers.values()
                              if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
                if not candidates:
                    return None
                return max(candidates, key=lambda s: s.free_capacity()).id

            def migration_decision(self, draining_server, servers):
                """
                *** رفع ناهماهنگی مهم: قبلاً اینجا [] برمی‌گشت (بدون migration
                واقعی)، در حالی‌که در PPOAlgorithm (زمان inference/ارزیابی)
                از منطق واقعی GreedyAlgorithm استفاده می‌شد. این یعنی عامل در
                آموزش یاد می‌گرفت TURN_OFF روی سروری با سرویس تک‌رپلیکا «امن»
                است (چون drain بدون migration خودکار لغو می‌شد - نگاه کنید
                simulator/engine.py:_start_server_drain)، ولی موقع ارزیابی
                واقعی همان اکشن رفتار متفاوتی داشت. حالا هر دو از همان منطق
                (GreedyAlgorithm.migration_decision) استفاده می‌کنند.
                """
                from algorithms.greedy.greedy_algorithm import GreedyAlgorithm
                return GreedyAlgorithm().migration_decision(draining_server, servers)

        self._impl = _Impl()

    def __getattr__(self, item):
        return getattr(self._impl, item)