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
 
_NORM_REJECTED_PER_TICK = 1.0


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
        non_noop_servers = [sid for sid, ptype in server_actions.items()
                             if ptype != ProvisionActionType.NO_CHANGE]
        if non_noop_servers:
            chosen_sid = int(self.np_random.choice(non_noop_servers))
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
        avg_dv_rate = (sum(s["deadline_violation_rate"] for s in active_svcs) / len(active_svcs)) if active_svcs else 0.0


        active_utils = [s["utilization"] for s in snapshot["servers"].values()
                         if s["state"] == ServerState.ACTIVE]
        load_cv = 0.0
        if len(active_utils) >= 2 and np.mean(active_utils) > 0:
            load_cv = float(np.std(active_utils) / np.mean(active_utils))
 
        norm_rt = min(g["avg_response_time_recent"] / NORM_RESPONSE_TIME_SEC, 2.0)
        norm_energy = min(g["energy_recent_joule"] / NORM_ENERGY_JOULE, 2.0)
        norm_lb = min(load_cv, 2.0) 
        norm_rejected = min(g["num_rejected_recent"] / _NORM_REJECTED_PER_TICK, 2.0)
  
        
        penalty = (w["w1_response_time"] * norm_rt +
                   w["w2_deadline"] * avg_dv_rate +
                   w["w3_energy"] * norm_energy +
                   w["w4_load_balance"] * norm_lb +
                   w["w5_rejected"] * norm_rejected)
        penalty += CFG.ppo_penalty_per_action * n_actions_applied
 
        self._last_reward_components = {
            "response_time": w["w1_response_time"] * norm_rt,
            "deadline": w["w2_deadline"] * avg_dv_rate,
            "energy": w["w3_energy"] * norm_energy,
            "load_balance": w["w4_load_balance"] * norm_lb,
            "rejected": w["w5_rejected"] * norm_rejected,
            "action_penalty": CFG.ppo_penalty_per_action * n_actions_applied,
        }
        return -float(penalty) 

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray: 
        masks = []
        now = self.engine.now
        for sid in _SERVICE_IDS:
            cooldown = (now - self.engine._service_last_scale_time[sid]) < CFG.cooldown_sec
            can_up = (not cooldown) and self._any_server_can_host(sid)
            reps = self.engine.replicas_by_service.get(sid, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            mature = [r for r in ready
                      if (now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
            can_down = (not cooldown) and len(ready) > 1 and len(mature) > 0
            masks.extend([True, can_up, can_down])  # NO_CHANGE همیشه مجاز است
        n_active = sum(1 for s in self.engine.servers.values() if s.state == ServerState.ACTIVE)
        for sid in _SERVER_IDS:
            s = self.engine.servers[sid]
            cooldown = s.in_cooldown(now, CFG.cooldown_sec)
            can_on = (s.state == ServerState.OFF) and (not cooldown)
            can_off = (s.state == ServerState.ACTIVE and not cooldown and n_active > 1
                       and (now - s.last_transition_time) >= CFG.min_active_duration_sec)
            masks.extend([True, can_on, can_off])
        return np.array(masks, dtype=bool)

    def _any_server_can_host(self, service_id: int) -> bool:
        cpu = CFG.services_info[service_id]["resource_mips"] 
        return any(s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)
                   for s in self.engine.servers.values())



class _MinimalSharedAlgorithm: 
    def __init__(self):
        from algorithms.base import AlgorithmBase

        class _Impl(AlgorithmBase):
            name = "ppo_shared"

            def scale_decision(self, *a, **k):
                raise NotImplementedError

            def provision_decision(self, *a, **k):
                raise NotImplementedError

            def select_placement_server(self, service_id, servers): 
                from common.models import ServerState
                from common.config import CFG as _CFG
                cpu = _CFG.services_info[service_id]["resource_mips"]
                candidates = [s for s in servers.values()
                              if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
                if not candidates:
                    return None
                return max(candidates, key=lambda s: s.free_capacity()).id

            def migration_decision(self, draining_server, servers): 
                from algorithms.greedy.greedy_algorithm import GreedyAlgorithm
                return GreedyAlgorithm().migration_decision(draining_server, servers)

        self._impl = _Impl()

    def __getattr__(self, item):
        return getattr(self._impl, item)