"""
algorithms/ppo/ppo_algorithm.py
پیاده‌سازی AlgorithmBase با مدل PPO آموزش‌دیده، برای استفاده در run() عادی
موتور (همان مسیری که Greedy/Voila/HPA از آن عبور می‌کنند) تا مقایسه‌ی
چهارگانه با معیارهای یکسان ممکن شود (بخش ۱۱.۵: ارزیابی inference-only).

*** نکته‌ی طراحی مهم: چون AlgorithmBase.scale_decision() سیگنیچرش شامل `now`
نیست (فقط service_id و metrics_snapshot)، ولی مدل PPO باید یک‌بار در هر تیک
(نه یک‌بار به ازای هر سرویس) پیش‌بینی انجام دهد، از این ترتیب فراخوانی
موتور (simulator/engine.py:_handle_decision_tick) استفاده می‌شود: همیشه
provision_decision() *قبل* از حلقه‌ی scale_decision() هر سرویس صدا زده
می‌شود؛ بنابراین پیش‌بینی مدل در provision_decision() یک‌بار محاسبه و کش
می‌شود و scale_decision() فقط از کش می‌خواند.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import random as _tie_break_random
from common.config import CFG
from common.models import Server, ServerState
from common.state_builder import build_state_vector
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep
from algorithms.greedy.greedy_algorithm import GreedyAlgorithm

from common.geo import haversine_km, network_delay_ms
_SERVICE_IDS = sorted(CFG.services_info.keys())
_SERVER_IDS = sorted(CFG.server_info.keys())
_SCALE_MAP = {0: ScaleAction.NO_CHANGE, 1: ScaleAction.SCALE_UP, 2: ScaleAction.SCALE_DOWN}
_PROVISION_MAP = {0: ProvisionActionType.NO_CHANGE, 1: ProvisionActionType.TURN_ON,
                   2: ProvisionActionType.TURN_OFF}


class PPOAlgorithm(AlgorithmBase):
    name = "ppo"
 
    def __init__(self, model_path, deterministic=True, latency_aware_routing=False , use_solver_placement = True, placement_weights=None): 
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as e:
            raise ImportError(
                "sb3-contrib نصب نیست. اجرا کنید: pip install -r requirements.txt"
            ) from e
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic
        # *** هماهنگ با algorithms/ppo/env.py:step - رفع بایاس به‌سمت
        # کمترین server_id، با تصادفی‌سازی بذردار (برای reproducibility
        # ارزیابی نهایی) به‌جای انتخاب قطعی اولین سرور.
        self._tie_break_rng = _tie_break_random.Random(CFG.seed)
        self._cached_tick_key: Optional[float] = None
        self._cached_scale: Dict[int, ScaleAction] = {}
        self._cached_provision = ProvisionAction(ProvisionActionType.NO_CHANGE)
        # قوانین مشترک غیر-یادگیرنده (placement/migration - خارج از فضای اکشن PPO، بخش ۱۱.۱)
        self._helper = GreedyAlgorithm()
        self.latency_aware_routing = latency_aware_routing

        # *** جای‌گذاری اولیه‌ی بهینه با ILP بر پایه‌ی داده‌ی آموزشی (نگاه کنید
        # algorithms/ppo/optimal_placement.py). فقط یک‌بار در طول عمر این
        # instance حل و کش می‌شود - چون همان چیزی است که یک engine واحد در
        # ابتدای اجرا یک‌بار initial_placement صدا می‌زند.
        self.use_solver_placement = use_solver_placement
        self._solver_selected_servers: Optional[List[int]] = None
 
        self._placement_weights = placement_weights or {"w_count": 1.0, "w_energy": 1.0, "w_distance": 1.0}
      
    # ------------------------------------------------------------------
    def initial_placement(self, servers: Dict[int, Server], active_bts):
        if not self.use_solver_placement:
            return super().initial_placement(servers, active_bts)

        if self._solver_selected_servers is None:
            self._solver_selected_servers = self._solve_from_training_data(servers, active_bts)

        return self._ensure_sufficient_capacity(servers, list(self._solver_selected_servers))


    def _solve_from_training_data(self, servers, active_bts_fallback):
        from algorithms.ppo.optimal_placement import aggregate_training_demand, solve_optimal_server_selection
        try:
            from data.loader import load_train
            train_events = load_train()
            demand_points = aggregate_training_demand(train_events)
            selected = solve_optimal_server_selection(servers, demand_points, **self._placement_weights)
            if selected:
                print(f"[PPO] جای‌گذاری اولیه‌ی چندهدفه حل شد: {len(selected)} سرور، "
                    f"سرورها: {sorted(selected)}")
                return selected
            print("[PPO] solver جواب قابل‌قبول پیدا نکرد؛ fallback به پوشش حریصانه‌ی مشترک.")
        except Exception as e:
            print(f"[PPO] حل ILP شکست خورد ({e}); fallback به پوشش حریصانه‌ی مشترک.")
        return super(PPOAlgorithm, self).initial_placement(servers, active_bts_fallback)

    @staticmethod
    def _ensure_sufficient_capacity(servers: Dict[int, Server], selected: List[int]) -> List[int]:
        """دقیقاً همان قید انتهای AlgorithmBase.initial_placement: اگر مجموع
        ظرفیت سرورهای انتخابی کمتر از نیاز کل ۱۵ سرویس بود، نزدیک‌ترین
        سرورهای باقی‌مانده را هم اضافه کن."""
        total_cpu_needed = sum(s["cpu_demand"] for s in CFG.services_info.values())
        if sum(servers[sid].capacity for sid in selected) >= total_cpu_needed:
            return selected
        remaining = [sid for sid in servers if sid not in selected]
        remaining.sort(key=lambda sid: min(
            haversine_km(servers[sid].lat, servers[sid].long, servers[s2].lat, servers[s2].long)
            for s2 in selected) if selected else 0)
        while sum(servers[sid].capacity for sid in selected) < total_cpu_needed and remaining:
            selected.append(remaining.pop(0))
        return selected
    # ------------------------------------------------------------------
    def _predict_and_cache(self, servers: Dict[int, Server], metrics_snapshot: dict, now: float):
        if self._cached_tick_key == now:
            return  # این تیک قبلاً پیش‌بینی شده
        obs = build_state_vector(metrics_snapshot, servers)
        action_masks = self._build_action_masks(servers, metrics_snapshot)
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=self.deterministic)

        self._cached_scale = {sid: _SCALE_MAP[int(action[i])] for i, sid in enumerate(_SERVICE_IDS)}
        non_noop = []
        for j, sid in enumerate(_SERVER_IDS):
            ptype = _PROVISION_MAP[int(action[len(_SERVICE_IDS) + j])]
            if ptype != ProvisionActionType.NO_CHANGE:
                non_noop.append((sid, ptype))
        provision = ProvisionAction(ProvisionActionType.NO_CHANGE)
        if non_noop:
            chosen_sid, chosen_ptype = self._tie_break_rng.choice(non_noop)
            provision = ProvisionAction(chosen_ptype, chosen_sid)
        self._cached_provision = provision
        self._cached_tick_key = now

    def _build_action_masks(self, servers: Dict[int, Server], snapshot: dict):
        """*** باید دقیقاً هم‌راستا با algorithms/ppo/env.py:action_masks()
        باشد - همان باگ (can_host بدون چک ACTIVE) اینجا هم بود، چون این
        تابع مسیر inference/ارزیابی نهایی است (نه فقط آموزش)."""
        import numpy as np
        masks = []
        for sid in _SERVICE_IDS:
            sv = snapshot["services"][sid]
            cpu = CFG.services_info[sid]["cpu_demand"]
            can_up = any(s.state == ServerState.ACTIVE and s.can_host(sid, cpu)
                         for s in servers.values())
            can_down = sv["n_ready_replicas"] > 1
            masks.extend([True, can_up, can_down])
        for sid in _SERVER_IDS:
            st = snapshot["servers"][sid]["state"]
            masks.extend([True, st == ServerState.OFF, st == ServerState.ACTIVE])
        return np.array(masks, dtype=bool)

    # ------------------------------------------------------------------
    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        return self._cached_scale.get(service_id, ScaleAction.NO_CHANGE)

    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        self._predict_and_cache(servers, metrics_snapshot, now)
        return self._cached_provision

    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]:
        # طبق بخش ۱۱.۳: «بیشترین ظرفیت آزاد» - این تصمیم بخشی از فضای اکشن یادگیرنده نیست
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.free_capacity()).id

    def migration_decision(self, draining_server: Server,
                            servers: Dict[int, Server]) -> List[MigrationStep]:
        return self._helper.migration_decision(draining_server, servers)

    def select_replica(self, request, candidate_replicas, servers, now):
        if not candidate_replicas:
            return None
        if not self.latency_aware_routing:
            return super().select_replica(request, candidate_replicas, servers, now)


        best, best_latency = None, float("inf")
        for r in candidate_replicas:
            if r.queue_occupancy(now) >= r.queue_len:
                continue  
            server = servers[r.server_id]
            distance_km = haversine_km(request.bts_lat, request.bts_long, server.lat, server.long)
            delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
            rtt_sec = 2 * delay_ms / 1000.0

            # تخمین واقعیِ زمان انتظار: دقیقاً همون چیزی که Replica.try_admit
            # استفاده می‌کنه (available_at) - نه یک heuristic بر پایه‌ی occupancy
            est_wait_sec = max(0.0, r.available_at - now)
            est_total_latency = rtt_sec + est_wait_sec + r.exec_time

            if est_total_latency < best_latency:
                best_latency = est_total_latency
                best = r

        return best        