"""
algorithms/ppo/env.py
محیط Gymnasium طبق بخش ۱۱ سند. هر گام env = یک DECISION_TICK از موتور
(بخش ۱۱.۱: عامل هر DECISION_INTERVAL_SEC سه نوع تصمیم می‌گیرد).

Action (بخش ۱۱.۳): MultiDiscrete
    ۱۵ بعد اول: برای هر سرویس {0=NO_CHANGE, 1=SCALE_UP, 2=SCALE_DOWN}
    ۱۰ بعد بعدی: برای هر سرور {0=NO_CHANGE, 1=TURN_ON, 2=TURN_OFF}

Reward (بخش ۱۱.۴): ترکیب وزن‌دار منفی از ۴ معیار + جریمه‌ی درخواست ردشده +
    جریمه‌ی ثابت کوچک هر اکشن (برای جلوگیری از flapping بی‌مورد عامل).

*** CHANGELOG (بازبینی ۳): جریمه‌ی «درخواست ردشده» قبلاً به‌صورت خام
(PPO_PENALTY_PER_REJECTED * num_rejected_recent) اضافه می‌شد - یعنی یک عدد
*نامحدود* و *نرمال‌نشده*، در حالی‌که ۴ جزء دیگر reward همه در بازه‌ی [۰,۲]
نرمال شده بودند. در تیک‌های شلوغ که چند درخواست هم‌زمان رد می‌شدند (مثلاً
۶-۱۰ تا)، این جمله می‌توانست ۵ تا ۱۵ برابر بزرگ‌تر از مجموع ۴ جزء دیگر
باشد و کل سیگنال reward را تحت‌الشعاع قرار دهد. نتیجه‌ی مشاهده‌شده (روی
Data4.csv): عامل یاد گرفت تقریباً هیچ اکشنی نزند (۵۵ اکشن در کل ۲۸۸۰ تیک،
در برابر ۵۰۵ تای Greedy) و فقط تعداد سرور فعال را از ابتدا بالا نگه دارد
(avg_active_servers=3.07 در برابر ~۲ بقیه) تا هرگز درخواستی رد نشود - که
باعث بدترین انرژی (۳۸.۱M ژول در برابر ۳۲-۳۵.۷M) و بدترین توازن بار
(cv=0.71 در برابر ۰.۲۸-۰.۵۸) در بین هر ۴ الگوریتم شد.

اصلاح: num_rejected_recent هم مثل بقیه نرمال و به [۰,۲] کلمپ می‌شود
(_NORM_REJECTED_PER_TICK به‌عنوان مقیاس کالیبراسیون - بخش ۱۳: قابل تنظیم)
و یک وزن صریح w5_rejected به آن اختصاص می‌یابد؛ دیگر PPO_PENALTY_PER_REJECTED
جداگانه استفاده نمی‌شود (نگاه کنید common/config.py برای وزن‌های جدید).

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

# *** بخش ۱۳ سند (قابل کالیبراسیون): مقیاس نرمال‌سازی num_rejected_recent.
# تأیید مجدد با calibrate_constants.py روی Data4.csv (بعد از فیکس‌های اخیر
# موتور مشترک): توزیع این متریک به‌شدت دم‌سنگین است (mean=0.25، ولی
# max=175) و فقط ۴.۵۱٪ تیک‌ها اصلاً رد شدنی دارند - یعنی p90/p95 روی کل
# تیک‌ها هر دو صفر می‌شوند (غیرقابل‌استفاده به‌عنوان مقیاس). به‌جای آن، از
# توزیع *شرطی* (فقط تیک‌هایی که واقعاً رد شدن داشته‌اند) استفاده شد:
# p50=1.0, p90=3.0, p95=10.3, mean=5.62. مقدار انتخابی نزدیک به mean شرطی
# است - جالب توجه که این تقریباً برابر مقدار قبلی (5.0) درآمد، یعنی طراحی
# اولیه با شهود درست انتخاب شده بود، هرچند بدون داده‌ی دقیق پشتوانه.
_NORM_REJECTED_PER_TICK = 6.0


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
        # *** بازبینی ۴: اجزای خام (نرمال‌شده ولی بدون وزن) آخرین محاسبه‌ی
        # reward، برای مانیتورینگ جداگانه در حین آموزش - نگاه کنید
        # algorithms/ppo/train.py:RewardComponentLoggingCallback. بدون این،
        # غالب‌شدن یک جزء (مثل انرژی - دقیقاً همان چیزی که باعث رگرسیون
        # «فروپاشی provisioning» شد) فقط بعد از اتمام کامل training و اجرای
        # evaluation/compare_runs قابل تشخیص بود.
        self._last_reward_components: dict | None = None

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
        # *** رفع بایاس سیستماتیک: قبلاً همیشه اولین سرور غیر-NO_CHANGE به
        # ترتیب id صعودی انتخاب می‌شد (چون dict/for ترتیب insertion=sorted
        # id را حفظ می‌کند) - یعنی اگر عامل هم‌زمان برای دو سرور مختلف
        # اکشن غیرصفر می‌زد، همیشه کمترین id برنده بود، صرف‌نظر از اهمیت
        # واقعی. این یک بایاس مصنوعی و ثابت در سیگنال یادگیری ایجاد می‌کرد.
        # حالا بین کاندیدهای هم‌زمان، با RNG خودِ محیط (بذردار، برای
        # reproducibility) به‌صورت یکنواخت انتخاب می‌شود.
        non_noop_servers = [sid for sid, ptype in server_actions.items()
                             if ptype != ProvisionActionType.NO_CHANGE]
        if non_noop_servers:
            chosen_sid = int(self.np_random.choice(non_noop_servers))
            provision_action = ProvisionAction(server_actions[chosen_sid], chosen_sid)
            n_actions_taken += 1
            
            
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
        # *** رفع مشکل غالب‌شدن reward توسط جریمه‌ی رد (نگاه کنید CHANGELOG بالا):
        # قبلاً این جمله نرمال نبود و می‌توانست ۵-۱۵ برابر بقیه‌ی اجزا شود.
        norm_rejected = min(g["num_rejected_recent"] / _NORM_REJECTED_PER_TICK, 2.0)

        penalty = (w["w1_response_time"] * norm_rt +
                   w["w2_deadline"] * avg_dv_rate +
                   w["w3_energy"] * norm_energy +
                   w["w4_load_balance"] * norm_lb +
                   w["w5_rejected"] * norm_rejected)
        penalty += CFG.ppo_penalty_per_action * n_actions_taken

        # *** بازبینی ۴: ثبت اجزای وزن‌دار (سهم واقعی هرکدام در penalty نهایی)
        # برای لاگ جداگانه - نگاه کنید __init__ برای دلیل.
        self._last_reward_components = {
            "response_time": w["w1_response_time"] * norm_rt,
            "deadline": w["w2_deadline"] * avg_dv_rate,
            "energy": w["w3_energy"] * norm_energy,
            "load_balance": w["w4_load_balance"] * norm_lb,
            "rejected": w["w5_rejected"] * norm_rejected,
            "action_penalty": CFG.ppo_penalty_per_action * n_actions_taken,
        }
        return -float(penalty) 

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """اینترفیس مورد انتظار sb3_contrib.common.wrappers.ActionMasker /
        MaskableMultiDiscrete: یک آرایه‌ی بولی مسطح به طول sum(nvec).

        *** رفع باگ عدم‌تطابق mask/اجرای واقعی: قبلاً can_up فقط can_host()
        (ظرفیت خام، بدون چک state) را می‌سنجید، در حالی‌که select_placement_server
        واقعی فقط سرورهای ACTIVE را کاندید می‌کند - یعنی mask گاهی SCALE_UP
        را "مجاز" نشان می‌داد در حالی‌که موقع اجرا با no_target_server شکست
        می‌خورد (روی سرور OFF/BOOTING/DRAINING با ظرفیت خالی). همچنین
        can_down از n_replicas (شامل STARTING) به n_ready_replicas تغییر
        کرد چون اجرای واقعی SCALE_DOWN فقط رپلیکاهای READY را می‌شمارد."""
        masks = []
        snapshot = self._last_snapshot
        for sid in _SERVICE_IDS:
            sv = snapshot["services"][sid]
            can_up = self._any_server_can_host(sid)
            can_down = sv["n_ready_replicas"] > 1
            masks.extend([True, can_up, can_down])  # NO_CHANGE همیشه مجاز است
        for sid in _SERVER_IDS:
            st = snapshot["servers"][sid]["state"]
            can_on = st == ServerState.OFF
            can_off = st == ServerState.ACTIVE
            masks.extend([True, can_on, can_off])
        return np.array(masks, dtype=bool)

    def _any_server_can_host(self, service_id: int) -> bool:
        cpu = CFG.services_info[service_id]["cpu_demand"]
        # *** فقط سرور ACTIVE می‌تواند واقعاً میزبان replica جدید شود -
        # هماهنگ با select_placement_server در _MinimalSharedAlgorithm/
        # سایر الگوریتم‌ها که همگی state == ACTIVE را شرط می‌گذارند.
        return any(s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)
                   for s in self.engine.servers.values())



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