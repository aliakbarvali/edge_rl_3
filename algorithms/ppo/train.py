"""
algorithms/ppo/train.py
طبق بخش ۱۱.۵ سند:
    - آموزش روی تایم‌لاین پیوسته‌ی شنبه‌های هفته ۱-۳ (data.loader.load_train()).
    - از Greedy به‌عنوان معلم برای BC warm-start استفاده می‌شود.
    - ارزیابی نهایی روی Data4.csv (بخش infer.py / evaluation/compare_runs.py)، جدا.

*** CHANGELOG (بازبینی ۲): سند بخش ۱۱.۵ صراحتاً می‌خواهد «لاگ منحنی یادگیری
(reward per episode) برای گزارش‌دهی» ذخیره شود. قبلاً Monitor(...) بدون
filename ساخته می‌شد (فقط stdout via verbose=1) و MaskablePPO بدون
tensorboard_log بود - یعنی هیچ فایل قابل‌رسم (نه CSV، نه TensorBoard) تولید
نمی‌شد. حالا:
  1) هر یک از n_envs محیط موازی، log خودش را در logs/monitor/env_{i}.monitor.csv
     می‌نویسد (ستون‌های r/l/t استاندارد Monitor - همان چیزی که برای رسم
     reward-per-episode لازم است).
  2) MaskablePPO با tensorboard_log=logs/tensorboard ساخته می‌شود؛ اگر
     tensorboard نصب نباشد sb3 خودش fallback می‌کند به فقط-CSV بدون کرش.

اجرا (بعد از pip install -r requirements.txt):
    python3 -m algorithms.ppo.train
    # برای دیدن منحنی یادگیری زنده (اختیاری، اگر tensorboard نصب باشد):
    tensorboard --logdir logs/tensorboard
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


MODEL_PATH = os.path.join(os.path.dirname(__file__), f"ppo_model_seed{CFG.seed}.zip")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
MONITOR_DIR = os.path.join(LOG_DIR, "monitor")
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")

_SCALE_TO_INT = {ScaleAction.NO_CHANGE: 0, ScaleAction.SCALE_UP: 1, ScaleAction.SCALE_DOWN: 2}
_PROVISION_TO_INT = {ProvisionActionType.NO_CHANGE: 0, ProvisionActionType.TURN_ON: 1,
                      ProvisionActionType.TURN_OFF: 2}

def model_path_for_seed(seed: int) -> str:
    return os.path.join(os.path.dirname(__file__), f"ppo_model_seed{seed}.zip")

MODEL_PATH = model_path_for_seed(CFG.seed)  



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
                               epochs: int = 10, batch_size: int = 64, lr: float = 1e-4,
                               log_path: str | None = None):
    """
    Warm-start سیاست با یادگیری تحت‌نظارت (cross-entropy) روی دموی Greedy،
    قبل از fine-tune با RL. مستقیماً روی model.policy (torch.nn.Module واقعی
    stable-baselines3) کار می‌کند.

    *** اگر log_path داده شود، loss هر epoch هم در یک CSV ذخیره می‌شود (بخش
    ۱۱.۵: «لاگ منحنی یادگیری ... برای گزارش‌دهی» - BC loss هم بخشی از همان
    منحنی یادگیری کامل است، نه فقط RL reward).
    """
    import torch
    import csv

    device = model.policy.device
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_arr, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = obs_t.shape[0]

    bc_log_rows = []
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
        avg_loss = total_loss / n
        print(f"[BC warm-start] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        bc_log_rows.append({"epoch": epoch + 1, "loss": avg_loss})

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
            writer.writeheader()
            writer.writerows(bc_log_rows)
        print(f"[BC warm-start] لاگ loss در {log_path} ذخیره شد.")


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


def main(total_timesteps: int = 3_000_000, bc_epochs: int = 25, window_hours: float = 3.0,
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

    os.makedirs(MONITOR_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    train_events = load_train()

    print("در حال جمع‌آوری دموی Greedy برای BC warm-start ...")
    demo_obs, demo_act = collect_greedy_demonstrations(train_events, max_ticks=bc_max_ticks)
    print(f"تعداد نمونه‌ی BC: {len(demo_obs)}")

    def mask_fn(env):
        return env.action_masks()

    def make_env(seed: int, env_idx: int):
        # *** هر محیط موازی provider تصادفی *مستقل* خودش را دارد (seed جدا)
        # تا پنجره‌های آموزشی هم‌بسته/تکراری بین محیط‌ها نشوند - این تنوع
        # داده‌ی هر batch را زیاد می‌کند و نوسان گرادیان را کم می‌کند.
        provider = make_random_window_provider(train_events, window_sec=window_hours * 3600, seed=seed)

        def _init():
            monitor_path = os.path.join(MONITOR_DIR, f"env_{env_idx}.monitor.csv")
            return Monitor(ActionMasker(EdgeResourceEnv(events_df_provider=provider), mask_fn),
                            filename=monitor_path)
        return _init

    # *** به‌جای ۱ محیط، n_envs محیط موازی (هم‌زمان، DummyVecEnv - نه
    # SubprocVecEnv، چون موتور شبیه‌سازی به‌قدری سریع است که سربار
    # multiprocessing ارزشش را ندارد و مشکلات pickling را هم دور می‌زند).
    vec_env = DummyVecEnv([make_env(CFG.seed + i, i) for i in range(n_envs)])
    # *** نرمال‌سازی خودکار reward با آمار running mean/std (به‌جای حدس دستی
    # ثابت‌های مقیاس که چند بار کالیبراسیونش اشتباه از آب درآمد).
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)

    model = MaskablePPO(
        "MlpPolicy", vec_env, verbose=1, policy_kwargs=PPO_POLICY_KWARGS,
        n_steps=2048, batch_size=256, gamma=0.99, learning_rate=3e-4, seed=CFG.seed,
        ent_coef=0.01,
        tensorboard_log=TENSORBOARD_DIR,
    )

    print("در حال BC warm-start ...")
    behavior_cloning_pretrain(model, demo_obs, demo_act, epochs=bc_epochs,
                               log_path=os.path.join(LOG_DIR, "bc_warmstart_loss.csv"))

    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1), save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="ppo_ckpt")

    print(f"در حال آموزش PPO (fine-tune با RL، {n_envs} محیط موازی، {total_timesteps} timestep) ...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True,
                tb_log_name="ppo_run", callback=checkpoint_cb)

    model.save(MODEL_PATH) 
    vec_env.save(MODEL_PATH.replace(".zip", "_vecnormalize.pkl"))
    print(f"مدل ذخیره شد: {MODEL_PATH}")
    print(f"آمار VecNormalize ذخیره شد: {MODEL_PATH.replace('.zip', '_vecnormalize.pkl')}")
    print(f"لاگ‌های reward-per-episode هر محیط: {MONITOR_DIR}/env_*.monitor.csv")
    print(f"لاگ TensorBoard: {TENSORBOARD_DIR}  (تماشا با: tensorboard --logdir {TENSORBOARD_DIR})")


if __name__ == "__main__":
    main()