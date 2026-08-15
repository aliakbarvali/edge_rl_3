"""
algorithms/ppo/env.py
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common.config import CFG
from common.models import ServerState, ReplicaState
from common.state_builder import (build_state_vector, STATE_DIM,
                                   NORM_RESPONSE_TIME_SEC, NORM_ENERGY_JOULE)
from algorithms.base import ScaleAction, ProvisionAction, ProvisionActionType
from simulator.engine import SimulationEngine

N_SERVICES = CFG.n_services
N_SERVERS = CFG.n_servers
_SERVICE_IDS = sorted(CFG.services_info.keys())
_SERVER_IDS = sorted(CFG.server_info.keys())

_SCALE_MAP = {0: ScaleAction.NO_CHANGE, 1: ScaleAction.SCALE_UP, 2: ScaleAction.SCALE_DOWN}
_PROVISION_MAP = {0: ProvisionActionType.NO_CHANGE, 1: ProvisionActionType.TURN_ON,
                   2: ProvisionActionType.TURN_OFF}

import os as _os
_NORM_REJECTED_PER_TICK = float(_os.environ.get("EOTCH_NORM_REJECTED_PER_TICK", "2.0"))


def _any_server_can_host(engine: SimulationEngine, last_snapshot: dict | None,
                          service_id: int) -> bool:
    cpu = CFG.services_info[service_id]["resource_mips"]
    centroid = None
    if last_snapshot is not None:
        centroid = last_snapshot["services"][service_id].get("demand_centroid")
    bts_lat, bts_long = centroid if centroid else (None, None)
    return any(s.state == ServerState.ACTIVE
               and s.can_host(service_id, cpu, bts_lat=bts_lat, bts_long=bts_long)
               for s in engine.servers.values())


def compute_action_masks(engine: SimulationEngine, last_snapshot: dict | None) -> np.ndarray:
    masks = []
    now = engine.now
    for sid in _SERVICE_IDS:
        cooldown = (now - engine._service_last_scale_time[sid]) < CFG.cooldown_sec
        can_up = (not cooldown) and _any_server_can_host(engine, last_snapshot, sid)
        reps = engine.replicas_by_service.get(sid, [])
        ready = [r for r in reps if r.state == ReplicaState.READY]
        mature = [r for r in ready
                  if (now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
        can_down = (not cooldown) and len(ready) > 1 and len(mature) > 0
        masks.extend([True, can_up, can_down])

    n_active = sum(1 for s in engine.servers.values() if s.state == ServerState.ACTIVE)
    for sid in _SERVER_IDS:
        s = engine.servers[sid]
        cooldown = s.in_cooldown(now, CFG.cooldown_sec)
        can_on = (s.state == ServerState.OFF) and (not cooldown)
        can_off = (s.state == ServerState.ACTIVE and not cooldown and n_active > 1
                   and (now - s.last_transition_time) >= CFG.min_active_duration_sec)
        masks.extend([True, can_on, can_off])
    return np.array(masks, dtype=bool)


class EdgeResourceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, events_df_provider, teacher_algorithm=None):
        super().__init__()
        self.events_df_provider = events_df_provider
        self._shared_algo = teacher_algorithm or _MinimalSharedAlgorithm()

        self.action_space = spaces.MultiDiscrete([3] * N_SERVICES + [3] * N_SERVERS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)

        self.engine: SimulationEngine | None = None
        self._last_snapshot = None
        self._last_reward_components: dict | None = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        events_df = self.events_df_provider()
        self.engine = SimulationEngine(events_df, self._shared_algo, "ppo_train")
        # *** رفع باگ ۱ (train/inference mismatch در select_placement_server):
        # وقتی self._shared_algo نمونه‌ای از _MinimalSharedAlgorithm است (حالت
        # پیش‌فرض چون algorithms/ppo/train.py:make_env هیچ teacher_algorithm پاس
        # نمی‌دهد)، engine تازه‌ساخته‌شده‌ی همین اپیزود به آن bind می‌شود تا
        # select_placement_server/migration_decision بتوانند از demand_centroid
        # واقعی (engine._service_demand_centroid، هر تیک در
        # simulator/engine.py:_build_metrics_snapshot به‌روزرسانی می‌شود) استفاده
        # کنند - دقیقاً همان تک‌منبع حقیقتی که algorithms/ppo/ppo_algorithm.py
        # (مسیر inference/k8s واقعی) از self._last_snapshot می‌خواند. بدون این،
        # فاز RL fine-tune (بعد از BC warm-start) با یک سیاست placement کاملاً
        # متفاوت (بدون آگاهی جغرافیایی، فقط بیشترین ظرفیت آزاد) آموزش می‌دید،
        # درحالی‌که خودِ compute_action_masks (پایین‌تر) با _any_server_can_host
        # کاملاً centroid-aware است - یعنی حتی خودِ فاز training هم بین «اکشن
        # مجاز طبق mask» و «قابل‌اجرا بودن واقعی» ناسازگار بود.
        if hasattr(self._shared_algo, "bind_engine"):
            self._shared_algo.bind_engine(self.engine)
        self.engine.prime()
        snapshot = self.engine.peek_snapshot()
        self._last_snapshot = snapshot
        obs = build_state_vector(snapshot, self.engine.servers)
        return obs, {}

    def step(self, action):
        service_actions = {sid: _SCALE_MAP[int(action[i])] for i, sid in enumerate(_SERVICE_IDS)}
        server_actions = {sid: _PROVISION_MAP[int(action[N_SERVICES + j])]
                           for j, sid in enumerate(_SERVER_IDS)}

        provision_action = ProvisionAction(ProvisionActionType.NO_CHANGE)
        turn_ons = sorted(sid for sid, ptype in server_actions.items()
                           if ptype == ProvisionActionType.TURN_ON)
        turn_offs = sorted(sid for sid, ptype in server_actions.items()
                            if ptype == ProvisionActionType.TURN_OFF)
        chosen_list = turn_ons or turn_offs
        if chosen_list:
            chosen_sid = chosen_list[0]
            provision_action = ProvisionAction(server_actions[chosen_sid], chosen_sid)

        external = {"provision": provision_action, "scale": service_actions}

        m = self.engine.metrics
        before_counts = (m.num_scale_up, m.num_scale_down, m.num_turn_on, m.num_turn_off)

        snapshot, done = self.engine.step(external_actions=external)

        if done:
            obs = np.zeros(STATE_DIM, dtype=np.float32)
            reward = 0.0
            terminated = True
        else:
            after_counts = (m.num_scale_up, m.num_scale_down, m.num_turn_on, m.num_turn_off)
            n_actions_applied = sum(a2 - a1 for a1, a2 in zip(before_counts, after_counts))
            obs = build_state_vector(snapshot, self.engine.servers)
            reward = self._compute_reward(snapshot, n_actions_applied)
            terminated = False

        self._last_snapshot = snapshot
        return obs, reward, terminated, False, {}

    # ------------------------------------------------------------------
    def _compute_reward(self, snapshot: dict, n_actions_applied: int) -> float:
        w = CFG.ppo_reward_weights
        g = snapshot["global"]

        active_svcs = [s for s in snapshot["services"].values() if s["recent_arrivals"] > 0]
        if active_svcs:
            total_arrivals = sum(s["recent_arrivals"] for s in active_svcs)
            weighted_dv_rate = (sum(s["deadline_violation_rate"] * s["recent_arrivals"]
                                     for s in active_svcs) / total_arrivals) if total_arrivals else 0.0
            unweighted_dv_rate = sum(s["deadline_violation_rate"] for s in active_svcs) / len(active_svcs)
            alpha = CFG.ppo_deadline_fairness_alpha
            avg_dv_rate = alpha * weighted_dv_rate + (1 - alpha) * unweighted_dv_rate
        else:
            weighted_dv_rate = 0.0
            unweighted_dv_rate = 0.0
            avg_dv_rate = 0.0

        active_utils = [s["utilization"] for s in snapshot["servers"].values()
                         if s["state"] == ServerState.ACTIVE]
        load_cv = 0.0
        if len(active_utils) >= 2 and np.mean(active_utils) > 0:
            load_cv = float(np.std(active_utils) / np.mean(active_utils))

        norm_rt = min(g["avg_response_time_recent"] / NORM_RESPONSE_TIME_SEC, 2.0)
        norm_energy = min(g["energy_recent_joule"] / NORM_ENERGY_JOULE, 2.0)
        norm_lb = min(load_cv, 2.0)
        norm_rejected = min(g["num_rejected_recent"] / _NORM_REJECTED_PER_TICK, 2.0)
        n_services_rejected = sum(1 for s in snapshot["services"].values()
                                if s["rejection_rate"] > 0)
        norm_rejected_services = min(n_services_rejected / CFG.n_services, 1.0)
     
        penalty = (w["w1_response_time"] * norm_rt +
           w["w2_deadline"] * avg_dv_rate +
           w["w3_energy"] * norm_energy +
           w["w4_load_balance"] * norm_lb +
           w["w5_rejected"] * norm_rejected +
           w["w6_rejected_service_spread"] * norm_rejected_services)
        penalty += CFG.ppo_penalty_per_action * n_actions_applied

        self._last_reward_components = {
            "response_time": w["w1_response_time"] * norm_rt,
            "deadline": w["w2_deadline"] * avg_dv_rate,
            "energy": w["w3_energy"] * norm_energy,
            "load_balance": w["w4_load_balance"] * norm_lb,
            "rejected": w["w5_rejected"] * norm_rejected,
            "rejected_service_spread": w["w6_rejected_service_spread"] * norm_rejected_services,
            "action_penalty": CFG.ppo_penalty_per_action * n_actions_applied,
            "deadline_weighted_raw": weighted_dv_rate,
            "deadline_unweighted_raw": unweighted_dv_rate,
        }
        return -float(penalty)

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        return compute_action_masks(self.engine, self._last_snapshot)


class _MinimalSharedAlgorithm:
    def __init__(self):
        from algorithms.base import AlgorithmBase

        class _Impl(AlgorithmBase):
            name = "ppo_shared"

            def __init__(self):
                # *** رفع باگ ۱: ارجاع به engine جاری، توسط EdgeResourceEnv.reset
                # (از طریق bind_engine روی کلاس بیرونی) ست می‌شود.
                self.engine = None

            def scale_decision(self, *a, **k):
                raise NotImplementedError

            def provision_decision(self, *a, **k):
                raise NotImplementedError

            def _demand_centroid_or_none(self, service_id):
                if self.engine is None:
                    return None
                return self.engine._service_demand_centroid.get(service_id)

            def select_placement_server(self, service_id, servers):
                from common.models import ServerState
                from common.config import CFG as _CFG
                from common.geo import haversine_km
                cpu = _CFG.services_info[service_id]["resource_mips"]
                centroid = self._demand_centroid_or_none(service_id)
                bts_lat, bts_long = centroid if centroid else (None, None)
                candidates = [s for s in servers.values()
                              if s.state == ServerState.ACTIVE
                              and s.can_host(service_id, cpu, bts_lat=bts_lat, bts_long=bts_long)]
                if not candidates:
                    return None
                if centroid is not None:
                    distances = {s.id: haversine_km(centroid[0], centroid[1], s.lat, s.long)
                                 for s in candidates}
                    min_dist = min(distances.values())
                    near_pool = [s for s in candidates if distances[s.id] <= min_dist + 5.0]
                    return max(near_pool, key=lambda s: s.free_capacity()).id
                return max(candidates, key=lambda s: s.free_capacity()).id

            def migration_decision(self, draining_server, servers):
                from algorithms.greedy.greedy_algorithm import GreedyAlgorithm
                helper = GreedyAlgorithm()
                if self.engine is not None:
                    helper._last_snapshot = {
                        "services": {
                            sid: {"demand_centroid": centroid}
                            for sid, centroid in self.engine._service_demand_centroid.items()
                        }
                    }
                return helper.migration_decision(draining_server, servers)

        self._impl = _Impl()

    def bind_engine(self, engine):
        self._impl.engine = engine

    def __getattr__(self, item):
        return getattr(self._impl, item)