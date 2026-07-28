"""
algorithms/ppo/train.py
طبق بخش ۱۱.۵ سند:
    - آموزش روی تایم‌لاین پیوسته‌ی شنبه‌های هفته ۱-۳ (data.loader.load_train()).
    - از Greedy به‌عنوان معلم برای BC warm-start استفاده می‌شود.
    - ارزیابی نهایی روی Data4.csv (بخش infer.py / evaluation/compare_runs.py)، جدا.

اجرا (بعد از pip install -r requirements.txt):
    python3 -m algorithms.ppo.train
"""

from __future__ import annotations
import os
import numpy as np

from common.config import CFG
from common.state_builder import build_state_vector
from algorithms.base import ScaleAction, ProvisionActionType
from algorithms.greedy.greedy_algorithm import GreedyAlgorithm
from algorithms.ppo.env import EdgeResourceEnv, _SERVICE_IDS, _SERVER_IDS
from simulator.engine import SimulationEngine

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ppo_model.zip")

_SCALE_TO_INT = {ScaleAction.NO_CHANGE: 0, ScaleAction.SCALE_UP: 1, ScaleAction.SCALE_DOWN: 2}
_PROVISION_TO_INT = {ProvisionActionType.NO_CHANGE: 0, ProvisionActionType.TURN_ON: 1,
                      ProvisionActionType.TURN_OFF: 2}


def _encode_action(decisions: dict) -> np.ndarray:
    """تصمیمات خام موتور (dict برگردانده‌شده از engine._last_tick_decisions) را
    به همان بردار MultiDiscrete که EdgeResourceEnv تولید می‌کند تبدیل می‌کند."""
    scale = decisions["scale"]
    provision = decisions["provision"]
    action = [_SCALE_TO_INT.get(scale.get(sid, ScaleAction.NO_CHANGE), 0) for sid in _SERVICE_IDS]
    for sid in _SERVER_IDS:
        if provision is not None and getattr(provision, "server_id", None) == sid:
            action.append(_PROVISION_TO_INT[provision.action])
        else:
            action.append(0)
    return np.array(action, dtype=np.int64)


def collect_greedy_demonstrations(events_df, max_ticks: int | None = None):
    """اجرای Greedy روی داده و ثبت (state, action) هر تیک برای BC warm-start."""
    algo = GreedyAlgorithm()
    engine = SimulationEngine(events_df, algo, "greedy_teacher")
    engine.prime()
    obs_list, act_list = [], []
    n = 0
    while True:
        snapshot, done = engine.step()  # external_actions=None -> Greedy تصمیم می‌گیرد
        if done:
            break
        obs_list.append(build_state_vector(snapshot, engine.servers))
        act_list.append(_encode_action(engine._last_tick_decisions))
        n += 1
        if max_ticks and n >= max_ticks:
            break
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def behavior_cloning_pretrain(model, obs_arr: np.ndarray, act_arr: np.ndarray,
                               epochs: int = 10, batch_size: int = 64, lr: float = 1e-4):
    """
    Warm-start سیاست با یادگیری تحت‌نظارت (cross-entropy) روی دموی Greedy،
    قبل از fine-tune با RL. مستقیماً روی model.policy (torch.nn.Module واقعی
    stable-baselines3) کار می‌کند.
    """
    import torch

    device = model.policy.device
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_arr, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = obs_t.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_obs, batch_act = obs_t[idx], act_t[idx]
            dist = model.policy.get_distribution(batch_obs)
            log_prob = dist.log_prob(batch_act)  # مجموع log-prob هر بعدِ MultiDiscrete
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"[BC warm-start] epoch {epoch + 1}/{epochs}  loss={total_loss / n:.4f}")


def make_random_window_provider(train_events, window_sec: float, seed: int = CFG.seed):
    """
    برای هر اپیزود آموزشی، یک پنجره‌ی زمانی تصادفی از تایم‌لاین سه‌روزه‌ی
    train انتخاب می‌کند (اپیزودهای کوتاه‌تر از کل ۳ روز -> آموزش سریع‌تر و
    تنوع بیشتر شرایط اولیه‌ی هر اپیزود).
    """
    rng = np.random.default_rng(seed)
    total_span = float(train_events.global_start_sec.max())

    def provider():
        start = float(rng.uniform(0, max(total_span - window_sec, 0)))
        window = train_events[(train_events.global_start_sec >= start) &
                               (train_events.global_start_sec < start + window_sec)].copy()
        window["global_start_sec"] -= start  # شروع اپیزود از ۰ (لازم برای جایگذاری اولیه‌ی موتور)
        return window.reset_index(drop=True)

    return provider


def main(total_timesteps: int = 500_000, bc_epochs: int = 10, window_hours: float = 3.0,
         bc_max_ticks: int | None = None, n_envs: int = 8):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as e:
        raise SystemExit(
            "sb3-contrib/stable-baselines3 نصب نیستند. اول اجرا کنید:\n"
            "    pip install -r requirements.txt\n"
            f"(خطای اصلی: {e})"
        )
    from algorithms.ppo.policy_network import PPO_POLICY_KWARGS
    from data.loader import load_train

    train_events = load_train()

    print("در حال جمع‌آوری دموی Greedy برای BC warm-start ...")
    demo_obs, demo_act = collect_greedy_demonstrations(train_events, max_ticks=bc_max_ticks)
    print(f"تعداد نمونه‌ی BC: {len(demo_obs)}")

    def mask_fn(env):
        return env.action_masks()

    def make_env(seed: int):
        # *** هر محیط موازی provider تصادفی *مستقل* خودش را دارد (seed جدا)
        # تا پنجره‌های آموزشی هم‌بسته/تکراری بین محیط‌ها نشوند - این تنوع
        # داده‌ی هر batch را زیاد می‌کند و نوسان گرادیان را کم می‌کند.
        provider = make_random_window_provider(train_events, window_sec=window_hours * 3600, seed=seed)

        def _init():
            return Monitor(ActionMasker(EdgeResourceEnv(events_df_provider=provider), mask_fn))
        return _init

    # *** به‌جای ۱ محیط، n_envs محیط موازی (هم‌زمان، DummyVecEnv - نه
    # SubprocVecEnv، چون موتور شبیه‌سازی به‌قدری سریع است که سربار
    # multiprocessing ارزشش را ندارد و مشکلات pickling را هم دور می‌زند).
    vec_env = DummyVecEnv([make_env(CFG.seed + i) for i in range(n_envs)])
    # *** نرمال‌سازی خودکار reward با آمار running mean/std (به‌جای حدس دستی
    # ثابت‌های مقیاس که چند بار کالیبراسیونش اشتباه از آب درآمد).
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)

    model = MaskablePPO(
        "MlpPolicy", vec_env, verbose=1, policy_kwargs=PPO_POLICY_KWARGS,
        n_steps=2048, batch_size=256, gamma=0.99, learning_rate=3e-4, seed=CFG.seed,
    )

    print("در حال BC warm-start ...")
    behavior_cloning_pretrain(model, demo_obs, demo_act, epochs=bc_epochs)

    print(f"در حال آموزش PPO (fine-tune با RL، {n_envs} محیط موازی، {total_timesteps} timestep) ...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(MODEL_PATH)
    vec_env.save(MODEL_PATH.replace(".zip", "_vecnormalize.pkl"))
    print(f"مدل ذخیره شد: {MODEL_PATH}")
    print(f"آمار VecNormalize ذخیره شد: {MODEL_PATH.replace('.zip', '_vecnormalize.pkl')}")


if __name__ == "__main__":
    main()