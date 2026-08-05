"""
algorithms/ppo/infer.py
اجرای مدل PPO آموزش‌دیده به‌صورت inference-only.

اجرا:
    python3 -m algorithms.ppo.infer
    python3 -m algorithms.ppo.infer --latency-aware-routing --w-energy 2.0
"""

from __future__ import annotations
import argparse
import json
import os

from common.logger import EventLogger
from algorithms.ppo.ppo_algorithm import PPOAlgorithm
from simulator.engine import SimulationEngine


def run_ppo_inference(events_df, model_path: str, log_path: str | None = None,
                       latency_aware_routing: bool = False, use_solver_placement: bool = True,
                       placement_weights: dict | None = None) -> dict:
    algo = PPOAlgorithm(model_path=model_path, deterministic=True,
                         latency_aware_routing=latency_aware_routing,
                         use_solver_placement=use_solver_placement,
                         placement_weights=placement_weights)
    logger = EventLogger(log_path, algorithm="ppo") if log_path else None
    engine = SimulationEngine(events_df, algo, "ppo", event_logger=logger)
    result = engine.run()
    if logger:
        logger.close()
    return result


def _parse_args():
    parser = argparse.ArgumentParser()
    # *** رفع باگ: قبلاً --seed اصلاً در CLI تعریف نشده بود، ولی پایین همین
    # فایل getattr(args, "seed", None) را می‌خواند - چون هیچ‌وقت چنین
    # attributeی وجود نداشت، همیشه None برمی‌گشت و بی‌صدا به CFG.seed
    # سقوط می‌کرد؛ کاربر هیچ راهی برای انتخاب seed از CLI نداشت.
    parser.add_argument("--seed", type=int, default=None,
                         help="اگر داده شود: مدل PPO مخصوص همین seed بارگذاری می‌شود و "
                              "نتایج در <output-dir>/seed<N>/ ذخیره می‌شوند (هماهنگ با "
                              "evaluation/compare_runs.py --seed و evaluation/aggregate_seeds.py)")
    parser.add_argument("--latency-aware-routing", action="store_true",
                         help="مسیریابی بر پایه‌ی تخمین کل تأخیر (شبکه+صف+اجرا)")
    parser.add_argument("--no-solver-placement", action="store_true",
                         help="غیرفعال‌کردن جای‌گذاری اولیه‌ی بهینه با ILP")
    parser.add_argument("--w-count", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-distance", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    from data.loader import load_test

    args = _parse_args()

    from algorithms.ppo.train import model_path_for_seed
    from common.config import CFG

    seed = args.seed or CFG.seed
    resolved_path = model_path_for_seed(seed)
    if not os.path.exists(resolved_path):
        raise SystemExit(
            f"مدل PPO برای seed={seed} پیدا نشد: {resolved_path}\n"
            f"اول اجرا کنید: python -m algorithms.ppo.train"
        )

    if args.seed is not None:
        # *** هماهنگ با evaluation/compare_runs.py --seed: زیرپوشه‌ی
        # خودکار seed<N> تا خروجی چند seed مختلف روی هم نوشته نشود و
        # مستقیماً قابل مصرف توسط evaluation/aggregate_seeds.py باشد.
        args.output_dir = os.path.join(args.output_dir, f"seed{args.seed}")

    test_events = load_test()
    os.makedirs(args.output_dir, exist_ok=True)
    result = run_ppo_inference(
        test_events,
        model_path=resolved_path,
        log_path=os.path.join(args.output_dir, "ppo_events.jsonl"),
        latency_aware_routing=args.latency_aware_routing,
        use_solver_placement=not args.no_solver_placement,
        placement_weights={"w_count": args.w_count, "w_energy": args.w_energy, "w_distance": args.w_distance},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    with open(os.path.join(args.output_dir, "ppo_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)