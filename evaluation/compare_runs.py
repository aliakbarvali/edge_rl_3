"""
evaluation/compare_runs.py
اجرای همه‌ی الگوریتم‌های *در دسترس* (پیاده‌شده و آماده) روی داده‌ی یکسان
(پیش‌فرض: Data4.csv طبق بخش ۱۱.۵) و تولید جدول مقایسه‌ی معیارهای بخش ۸.
الگوریتم‌های فاز ۲ (Voila/HPA) و PPO بدون مدل آموزش‌دیده به‌طور خودکار با
هشدار رد می‌شوند، نه با خطا - تا این اسکریپت همیشه با هر زیرمجموعه‌ای از
الگوریتم‌های آماده قابل اجرا باشد.

اجرا:
    python3 -m evaluation.compare_runs [--data test|train]
"""

from __future__ import annotations
import argparse
import os
import json
import pandas as pd

from common.logger import EventLogger
from simulator.engine import SimulationEngine


def _try_build(name: str):
    try:
        if name == "greedy":
            from algorithms.greedy.greedy_algorithm import GreedyAlgorithm
            return GreedyAlgorithm()
        if name == "voila":
            from algorithms.voila.voila_algorithm import VoilaAlgorithm
            return VoilaAlgorithm()
        if name == "hpa":
            from algorithms.hpa.hpa_algorithm import HPAAlgorithm
            return HPAAlgorithm()
        if name == "ppo":
            from algorithms.ppo.ppo_algorithm import PPOAlgorithm
            from algorithms.ppo.train import MODEL_PATH
            if not os.path.exists(MODEL_PATH):
                print(f"[رد شد] ppo: مدل آموزش‌دیده در {MODEL_PATH} پیدا نشد "
                      f"(python3 -m algorithms.ppo.train را اجرا کنید)")
                return None
            return PPOAlgorithm(model_path=MODEL_PATH)
    except (ImportError, NotImplementedError) as e:
        print(f"[رد شد] {name}: {e}")
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default="outputs")
    
    args = parser.parse_args()

    from data.loader import load_train, load_test
    events = load_train() if args.data == "train" else load_test()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for name in ["greedy", "voila", "hpa", "ppo"]:
        algo = _try_build(name)
        if algo is None:
            continue
        print(f"در حال اجرای {name} ...")
        logger = EventLogger(os.path.join(args.output_dir, f"{name}_events.jsonl"), algorithm=name)
        engine = SimulationEngine(events, algo, name, event_logger=logger)
        try:
            result = engine.run()
        except NotImplementedError as e:
            print(f"[رد شد] {name}: {e}")
            logger.close()
            continue
        logger.close()
        with open(os.path.join(args.output_dir, f"{name}_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        rows.append(result)

    if not rows:
        print("هیچ الگوریتمی آماده‌ی اجرا نبود.")
        return

    df = pd.DataFrame(rows)
    print("\n" + df.T.to_string())
    df.to_csv(os.path.join(args.output_dir, "comparison_summary.csv"), index=False)
    print(f"\nجدول مقایسه ذخیره شد: {args.output_dir}/comparison_summary.csv")


if __name__ == "__main__":
    main()
