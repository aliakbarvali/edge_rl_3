# algorithms/ppo/optimal_placement.py
"""
 
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from common.config import CFG
from common.geo import haversine_km


def aggregate_training_demand(train_events_df) -> List[Tuple[float, float, int]]: 
    counts = train_events_df.groupby(["Lat", "Long"]).size()
    return [(float(lat), float(lon), int(w)) for (lat, lon), w in counts.items()]


def solve_optimal_server_selection(
    servers: Dict,
    demand_points: List[Tuple[float, float, int]],
    l0_ms: float | None = None,
    min_total_capacity: int | None = None,
    w_count: float = 1.0,
    w_energy: float = 1.0,
    w_distance: float = 1.0,
    time_limit_sec: Optional[float] = 120.0,
) -> List[int]:
 
    try:
        import pulp
    except ImportError as e:
        raise ImportError("کتابخانه‌ی pulp نصب نیست. اجرا کنید: pip install pulp") from e

    l0_ms = l0_ms if l0_ms is not None else CFG.l0_ms
    if min_total_capacity is None:
        min_total_capacity = sum(s["resource_mips"] for s in CFG.services_info.values())
    server_ids = list(servers.keys())
    n_points = len(demand_points)
 
    coverage: Dict[int, set] = {}
    dist_km: Dict[Tuple[int, int], float] = {}
    for sid, s in servers.items():
        covered = set()
        for idx, (lat, lon, _w) in enumerate(demand_points):
            d_km = haversine_km(lat, lon, s.lat, s.long)
            delay = CFG.base_latency_ms + CFG.k_ms_per_km * d_km
            if delay <= l0_ms:
                covered.add(idx)
                dist_km[(idx, sid)] = d_km
        coverage[sid] = covered

    coverable = set().union(*coverage.values()) if coverage else set()
    if not coverable:
        return []

    prob = pulp.LpProblem("optimal_initial_placement", pulp.LpMinimize)
    y = {sid: pulp.LpVariable(f"y_{sid}", cat="Binary") for sid in server_ids} 
    x = {}
    for idx in coverable:
        for sid in server_ids:
            if idx in coverage[sid]:
                x[(idx, sid)] = pulp.LpVariable(f"x_{idx}_{sid}", cat="Binary")
 
    for idx in coverable:
        prob += pulp.lpSum(x[(idx, sid)] for sid in server_ids if (idx, sid) in x) == 1
 
    for (idx, sid) in x:
        prob += x[(idx, sid)] <= y[sid]
 
    prob += pulp.lpSum(servers[sid].capacity * y[sid] for sid in server_ids) >= min_total_capacity
 
    n_servers_total = len(server_ids)
    total_p_idle = sum(_p_idle_of(servers, sid) for sid in server_ids) 
    max_possible_dist = max(dist_km.values()) if dist_km else 1.0
    total_weight = sum(w for _, _, w in demand_points) or 1

    term_count = pulp.lpSum(y[sid] for sid in server_ids) / n_servers_total

    term_energy = pulp.lpSum(_p_idle_of(servers, sid) * y[sid] for sid in server_ids) / max(total_p_idle, 1e-9)

    term_distance = pulp.lpSum(
        demand_points[idx][2] * dist_km[(idx, sid)] * x[(idx, sid)]
        for (idx, sid) in x
    ) / (max_possible_dist * total_weight)

    prob += w_count * term_count + w_energy * term_energy + w_distance * term_distance

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]

    if status not in ("Optimal", "Not Solved"):
        return []

    selected = [sid for sid in server_ids if pulp.value(y[sid]) is not None and pulp.value(y[sid]) > 0.5]
    return selected if selected else []


def _p_idle_of(servers: Dict, sid: int) -> float:
    return servers[sid].p_idle