"""
آموزش عامل PPO روی FogServiceScalingEnv.

اجرا (از ریشه‌ی پروژه، بعد از pip install -r requirements.txt):
    python3 -m algorithms.ppo_rl.train_ppo

مدل آموزش‌دیده در algorithms/ppo_rl/ppo_model.zip ذخیره می‌شود و توسط
algorithms/ppo_rl/ppo_scaler.py برای استنتاج (inference) در main.py بارگذاری
می‌شود.

*** نکته: چون هر اپیزود یک سرویس تصادفی از ۱۵ سرویس را شبیه‌سازی می‌کند
(نگاه کنید به FogServiceScalingEnv)، این یک سیاست مشترک (generalist) است که
باید برای هر ۱۵ سرویس کار کند - نه ۱۵ مدل جداگانه. اگر نتایج برای سرویس‌های
سنگین (اجرای طولانی) ضعیف بود، می‌توانید per-service fine-tune کنید یا
observation را غنی‌تر کنید.
"""

from __future__ import annotations
import os
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    raise SystemExit(
        "stable-baselines3 نصب نیست. اول اجرا کنید:\n"
        "    pip install -r requirements.txt\n"
        f"(خطای اصلی: {e})"
    )

from common.config import CFG
from common.data_loader import prepare_dataset
from common.latency import build_latency_matrix, haversine_matrix
from common.qos import build_test_matrix
from algorithms.ppo_rl.ppo_env import FogServiceScalingEnv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ppo_model.zip")


def make_env():
    gateways, servers, events, service_profile = prepare_dataset()
    L = build_latency_matrix(gateways.Lat.values, gateways.Long.values,
                              servers.Lat.values, servers.Long.values)
    D_km = haversine_matrix(gateways.Lat.values, gateways.Long.values,
                             servers.Lat.values, servers.Long.values)
    T = build_test_matrix(L)
    env = FogServiceScalingEnv(gateways, servers, events, L, D_km, T, service_profile)
    return env


def main(total_timesteps: int = 200_000):
    env = Monitor(make_env())
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=CFG.N_CYCLES * 4,     # چند اپیزود کامل قبل از هر آپدیت
        batch_size=CFG.N_CYCLES,
        gamma=0.99,
        learning_rate=3e-4,
        seed=CFG.SEED,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(MODEL_PATH)
    print(f"مدل ذخیره شد: {MODEL_PATH}")


if __name__ == "__main__":
    main()
