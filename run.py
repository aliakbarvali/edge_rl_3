"""
run.py

"""

from __future__ import annotations
import argparse
import json
import os

from common.logger import EventLogger
from simulator.engine import SimulationEngine


def build_algorithm(name: str, args):
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
        from algorithms.ppo.train import model_path_for_seed
        from common.config import CFG
            
        seed = getattr(args, "seed", None) or CFG.seed
        resolved_path = model_path_for_seed(seed)
        if not os.path.exists(resolved_path):
            raise SystemExit(
                f"مدل PPO برای seed={seed} پیدا نشد: {resolved_path}\n"
                f"اول اجرا کنید: python -m algorithms.ppo.train"
            ) 
        return PPOAlgorithm(
            model_path=MODEL_PATH,
            latency_aware_routing=args.latency_aware_routing,
            use_solver_placement=not args.no_solver_placement,
        )
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
    
    parser.add_argument("--latency-aware-routing", action="store_true",
                         help="PPO: مسیریابی بر پایه‌ی تخمین کل تأخیر (شبکه+صف+اجرا) به‌جای صرفاً فاصله‌ی جغرافیایی")
    parser.add_argument("--no-solver-placement", action="store_true",
                         help="PPO: غیرفعال‌کردن جای‌گذاری اولیه‌ی بهینه با ILP (fallback به پوشش حریصانه‌ی مشترک همه‌ی الگوریتم‌ها)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    events = load_data(args.data)
    algorithm = build_algorithm(args.algorithm, args)
    
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
