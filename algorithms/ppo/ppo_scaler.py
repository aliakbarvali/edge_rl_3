"""
PPOScaler - رابط استنتاج (inference) برای عامل PPO آموزش‌دیده، با همان امضای
decide(gateway_load, allowed_servers) -> placement که Greedy/HPA/Voila دارند
(نگاه کنید به algorithms/{greedy,hpa,voila}) تا در main.py و common/simulator.py
بدون هیچ تغییری جایگزین آن‌ها شود.

پیش‌نیاز: قبل از استفاده باید یک مدل آموزش‌دیده وجود داشته باشد:
    pip install -r requirements.txt
    python3 -m algorithms.ppo_rl.train_ppo
"""

from __future__ import annotations
import os
import numpy as np

from common.config import CFG
from algorithms.ppo_rl.ppo_env import build_observation

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ppo_model.zip")


class PPOScaler:
    name = "PPO (RL)"

    def __init__(self, L: np.ndarray, T: np.ndarray, service_id: int, service_profile: dict,
                 model_path: str = MODEL_PATH):
        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 نصب نیست. اجرا کنید: pip install -r requirements.txt"
            ) from e
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"مدل PPO پیدا نشد: {model_path}\n"
                "اول آموزش بدهید: python3 -m algorithms.ppo_rl.train_ppo"
            )
        self.model = PPO.load(model_path)
        self.L, self.T = L, T
        self.n_servers = L.shape[1]
        self.n_gateways = L.shape[0]
        self.profile = service_profile[service_id]
        self.placement = np.zeros(self.n_servers, dtype=bool)
        self.cycle = 0  # *** شمارنده‌ی داخلی چرخه؛ فرض می‌شود decide() دقیقاً یک‌بار در هر چرخه صدا زده می‌شود

    def decide(self, gateway_load: np.ndarray, allowed_servers: np.ndarray | None = None) -> np.ndarray:
        if allowed_servers is None:
            allowed_servers = np.ones(self.n_servers, dtype=bool)

        obs = build_observation(allowed_servers, self.placement, gateway_load, self.L, self.T,
                                 self.profile, self.n_gateways, self.cycle)
        action, _ = self.model.predict(obs, deterministic=True)
        self.placement = np.asarray(action, dtype=bool) & allowed_servers

        self.cycle += 1
        return self.placement
