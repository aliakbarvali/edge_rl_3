"""
algorithms/ppo/infer.py
اجرای مدل PPO آموزش‌دیده به‌صورت inference-only (بدون یادگیری آنلاین) روی
داده (طبق بخش ۱۱.۵: ارزیابی روی Data4.csv برای مقایسه‌ی منصفانه با سه
الگوریتم قاعده‌محور).

اجرا:
    python3 -m algorithms.ppo.infer
"""

from __future__ import annotations
import json
import os

from common.logger import EventLogger
from algorithms.ppo.ppo_algorithm import PPOAlgorithm
from algorithms.ppo.train import MODEL_PATH
from simulator.engine import SimulationEngine


def run_ppo_inference(events_df, model_path: str = MODEL_PATH, log_path: str | None = None) -> dict:
    algo = PPOAlgorithm(model_path=model_path, deterministic=True)
    logger = EventLogger(log_path, algorithm="ppo") if log_path else None
    engine = SimulationEngine(events_df, algo, "ppo", event_logger=logger)
    result = engine.run()
    if logger:
        logger.close()
    return result


if __name__ == "__main__":
    from data.loader import load_test

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"مدل PPO پیدا نشد: {MODEL_PATH}\nاول آموزش بدهید: python3 -m algorithms.ppo.train")

    test_events = load_test()
    result = run_ppo_inference(test_events, log_path="outputs/ppo_events.jsonl")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ppo_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
