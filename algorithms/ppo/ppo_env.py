"""
محیط یادگیری تقویتی (Gymnasium) برای آموزش عامل PPO به‌عنوان یک service scaler.

*** فرض ساده‌کننده‌ی صریح (برای گزارش نهایی حتماً ذکر شود):
در دنیای واقعی پروژه (main.py)، هر ۱۵ سرویس هم‌زمان به یک ServerScaler مشترک
درخواست می‌دهند و ready_mask نتیجه‌ی اتحاد نیاز همه‌ی آن‌هاست. آموزش یک عامل
مشترک برای همه‌ی سرویس‌ها به‌طور هم‌زمان (multi-agent) پیچیدگی زیادی دارد؛
اینجا برای سادگی و امکان آموزش عملی، هر اپیزود فقط **یک سرویس** را شبیه‌سازی
می‌کند و به آن یک ServerScaler اختصاصی (نه مشترک) می‌دهد. یعنی عامل طوری
آموزش می‌بیند که انگار آن سرویس تنها مصرف‌کننده‌ی زیرساخت است.
در main.py (استقرار واقعی)، PPOScaler هرچه allowed_servers واقعی (از
ServerScaler مشترک) دریافت کند همان را به‌عنوان محدودیت اعمال می‌کند - پس
فقط رفتار سیاست ممکن است با آنچه در آموزش دیده کمی فرق کند (sim-to-real gap
معمول در RL)؛ برای کاهش این فاصله می‌توانید چند سرویس دیگر را هم به‌عنوان
"نویز پس‌زمینه‌ی بار" به Dispatcher اضافه کنید (نگاه کنید به پارامتر
`background_service_ids`).

State/Action/Reward:
    action  : MultiBinary(n_servers) - کدام سرورها replica این سرویس را میزبانی کنند
              (فقط سرورهای ready معتبرند؛ بقیه به‌صورت خودکار صفر می‌شوند)
    obs     : بردار ثابت شامل، برای هر سرور: (روشن است؟, replica رویش هست؟،
              فاصله‌ی latency وزن‌دار تا تقاضای این چرخه) + چند ویژگی سراسری
              (نسبت گیت‌وی فعال، بار کل نرمال‌شده، درصد پوشش latency فعلی،
              execution_time/resource همین سرویس، فاز روز)
    reward  : -(W_QOS * E%/100) - (W_RESOURCE * تعداد replica / n_servers)
              (می‌توانید W_QOS/W_RESOURCE را برای موازنه‌ی QoS در برابر مصرف
              منابع عوض کنید - دقیقاً همان trade-off که در README دیده‌اید)
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common.config import CFG
from common.data_loader import cycle_loads_for_service
from dispatcher.dispatcher import Dispatcher
from server_scaler.server_scaler import ServerScaler

W_QOS = 0.8        # وزن جریمه‌ی E% در reward
W_RESOURCE = 0.2   # وزن جریمه‌ی تعداد replica در reward


def build_observation(ready: np.ndarray, placement: np.ndarray, load: np.ndarray,
                       L: np.ndarray, T: np.ndarray, profile: dict,
                       n_gateways: int, cycle: int) -> np.ndarray:
    """
    ساخت بردار observation - هم توسط FogServiceScalingEnv (آموزش) و هم توسط
    PPOScaler (استنتاج در main.py) استفاده می‌شود تا این دو هرگز از هم جدا
    نیفتند (اگر فرمول obs جدا نگه داشته شود، مدل آموزش‌دیده در استقرار واقعی
    درست کار نخواهد کرد).
    """
    n_servers = len(ready)
    active_idx = np.where(load > 0)[0]

    if len(active_idx):
        w = load[active_idx]
        latency_to_demand = (L[active_idx] * w[:, None]).sum(axis=0) / w.sum()
    else:
        latency_to_demand = np.full(n_servers, CFG.LATENCY_MAX_MS)

    per_server = np.concatenate([
        ready.astype(np.float32),
        placement.astype(np.float32),
        (latency_to_demand / CFG.LATENCY_MAX_MS).astype(np.float32),
    ])

    coverage_frac = 0.0
    if len(active_idx) and placement.any():
        placed_idx = np.where(placement)[0]
        covered = np.any(T[np.ix_(active_idx, placed_idx)] == 1, axis=1)
        coverage_frac = float(covered.mean())

    global_feats = np.array([
        len(active_idx) / max(n_gateways, 1),
        load.sum() / 50.0,
        coverage_frac,
        profile["execution_time_sec"] / 150.0,
        profile["resource"] / 20.0,
        min(cycle, CFG.N_CYCLES - 1) / CFG.N_CYCLES,
    ], dtype=np.float32)

    return np.concatenate([per_server, global_feats]).astype(np.float32)


class FogServiceScalingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, gateways, servers, events, L: np.ndarray, D_km: np.ndarray,
                 T: np.ndarray, service_profile: dict, service_ids: list[int] | None = None,
                 co_per_cycle: float = CFG.CO_PER_CYCLE, seed: int | None = None):
        super().__init__()
        self.n_servers = len(servers)
        self.n_gateways = len(gateways)
        self.L, self.D_km, self.T = L, D_km, T
        self.service_profile = service_profile
        self.service_ids = list(service_ids or CFG.ACTIVE_SERVICES)
        self.co_per_cycle = co_per_cycle
        self.dispatcher = Dispatcher(L, D_km, CFG.LO_MS, service_profile)

        # پیش‌محاسبه‌ی بار هر سرویس (برای ساخت obs سریع، بدون فیلتر مکرر روی events)
        self._loads_cache = {sid: cycle_loads_for_service(events, sid, self.n_gateways)
                              for sid in self.service_ids}
        self.events = events

        self.action_space = spaces.MultiBinary(self.n_servers)
        obs_dim = self.n_servers * 3 + 6
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self._np_random_seed = seed
        self.service_id = None
        self.server_scaler = None
        self.placement = None
        self.cycle = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.service_id = int(self.np_random.choice(self.service_ids))
        self.loads = self._loads_cache[self.service_id]
        self.profile = self.service_profile[self.service_id]
        self.server_scaler = ServerScaler(self.n_servers)
        self.placement = np.zeros(self.n_servers, dtype=bool)
        self.cycle = 0
        return self._build_obs(), {"service_id": self.service_id}

    def step(self, action):
        ready = self.server_scaler.ready_mask()
        placement = np.asarray(action, dtype=bool) & ready  # سرور خاموش هرگز کاندید نیست
        self.placement = placement

        cyc_events = self.events[(self.events.cycle == self.cycle) &
                                  (self.events.ServiceID == self.service_id)]
        outcomes, replica_load = self.dispatcher.process_cycle(
            cyc_events, {self.service_id: placement}, ready, {self.service_id: self.co_per_cycle})

        n_total = len(outcomes)
        e_pct = 100.0 * outcomes.slow.sum() / n_total if n_total else 0.0
        n_replicas = int(placement.sum())
        reward = -(W_QOS * e_pct / 100.0) - (W_RESOURCE * n_replicas / self.n_servers)

        self.server_scaler.update(replica_load[self.service_id])

        self.cycle += 1
        terminated = self.cycle >= CFG.N_CYCLES
        obs = self._build_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"E_pct": e_pct, "n_replicas": n_replicas, "service_id": self.service_id}
        return obs, float(reward), terminated, False, info

    def _build_obs(self) -> np.ndarray:
        ready = self.server_scaler.ready_mask()
        cyc = min(self.cycle, CFG.N_CYCLES - 1)
        return build_observation(ready, self.placement, self.loads[cyc], self.L, self.T,
                                  self.profile, self.n_gateways, self.cycle)
