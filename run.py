"""
run.py
نقطه‌ی ورود اصلی، طبق بخش ۱۰ سند:
    python3 run.py --algorithm {greedy,voila,hpa,ppo} --mode {sim,k8s} [--data test|train]

این فایل فقط بر اساس --algorithm نمونه‌ی مناسب از AlgorithmBase می‌سازد و به
موتور شبیه‌سازی می‌دهد؛ خودش هیچ منطق تصمیم‌گیری‌ای ندارد.
"""

from __future__ import annotations
import argparse
import json
import os

from common.logger import EventLogger
from simulator.engine import SimulationEngine


def build_algorithm(name: str):
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
            raise SystemExit(f"مدل PPO پیدا نشد: {MODEL_PATH}\n"
                              f"اول آموزش بدهید: python3 -m algorithms.ppo.train")
        return PPOAlgorithm(model_path=MODEL_PATH)
    raise ValueError(f"الگوریتم ناشناخته: {name}")


def load_data(which: str):
    from data.loader import load_train, load_test
    return load_train() if which == "train" else load_test()


def main():
    parser = argparse.ArgumentParser(description="اجرای یک الگوریتم مدیریت منابع لبه")
    parser.add_argument("--algorithm", required=True, choices=["greedy", "voila", "hpa", "ppo"])
    parser.add_argument("--mode", default="sim", choices=["sim", "k8s"])
    parser.add_argument("--data", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    events = load_data(args.data)
    algorithm = build_algorithm(args.algorithm)
    logger = EventLogger(os.path.join(args.output_dir, f"{args.algorithm}_events.jsonl"),
                          algorithm=args.algorithm)

    if args.mode == "k8s":
        import asyncio
        from k8s_adapter.realtime_dispatcher import RealtimeEngine
        engine = RealtimeEngine(events, algorithm, args.algorithm, event_logger=logger)
        result = asyncio.run(engine.run())
    else:
        engine = SimulationEngine(events, algorithm, args.algorithm, event_logger=logger)
        result = engine.run()

    logger.close()
    
    

    out_path = os.path.join(args.output_dir, f"{args.algorithm}_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nنتیجه ذخیره شد: {out_path}")


if __name__ == "__main__":
    main()
