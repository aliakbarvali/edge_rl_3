from __future__ import annotations
import argparse
import os
import json
import pandas as pd

from common.logger import EventLogger
from simulator.engine import SimulationEngine


def _try_build(name: str, ppo_args=None):
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
            from algorithms.ppo.train import model_path_for_seed
            from common.config import CFG
                 
            seed = (ppo_args or {}).get("seed") or CFG.seed
            resolved_path = model_path_for_seed(seed)
            if not os.path.exists(resolved_path):
            
                raise FileNotFoundError(
                    f"مدل PPO برای seed={seed} پیدا نشد: {resolved_path}\n"
                    f"اول اجرا کنید: python -m algorithms.ppo.train"
                )
            ppo_args = ppo_args or {} 
            return PPOAlgorithm(
                model_path=resolved_path,
                latency_aware_routing=ppo_args.get("latency_aware_routing", False),
                use_solver_placement=ppo_args.get("use_solver_placement", True),
                placement_weights=ppo_args.get("placement_weights"),
            )
    except (ImportError, NotImplementedError, FileNotFoundError) as e:
        print(f"[رد شد] {name}: {e}")
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default="outputs") 
    parser.add_argument("--seed", type=int, default=None,
                         help="اگر داده شود: مدل PPO مخصوص همین seed بارگذاری می‌شود و "
                              "نتایج در <output-dir>/seed<N>/ ذخیره می‌شوند (برای "
                              "evaluation/aggregate_seeds.py)")
   
    parser.add_argument("--latency-aware-routing", action="store_true")
    parser.add_argument("--no-solver-placement", action="store_true")
    parser.add_argument("--w-count", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-distance", type=float, default=1.0)

    args = parser.parse_args()

    if args.seed is not None: 
        args.output_dir = os.path.join(args.output_dir, f"seed{args.seed}")

    ppo_args = {
        "seed": args.seed,
        "latency_aware_routing": args.latency_aware_routing,
        "use_solver_placement": not args.no_solver_placement,
        "placement_weights": {"w_count": args.w_count, "w_energy": args.w_energy,
                               "w_distance": args.w_distance},
    }

    from data.loader import load_train, load_test
    events = load_train() if args.data == "train" else load_test()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for name in ["greedy", "voila", "hpa", "ppo"]:
        algo = _try_build(name, ppo_args=ppo_args)
        if algo is None:
            continue
        print(f"در حال اجرای {name} ...",args.output_dir)
        
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
    #print("\n" + df.T.to_string())
    df.to_csv(os.path.join(args.output_dir, "comparison_summary.csv"), index=False)
    print(f"\nجدول مقایسه ذخیره شد: {args.output_dir}/comparison_summary.csv")


if __name__ == "__main__":
    main()