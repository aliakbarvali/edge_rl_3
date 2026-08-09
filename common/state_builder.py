"""
common/state_builder.py
 
"""

from __future__ import annotations
import numpy as np

from common.config import CFG
from common.models import ServerState

STATE_DIM = CFG.n_servers * 6 + CFG.n_services * 6 + 2  

_SERVER_STATE_ORDER = [ServerState.OFF, ServerState.BOOTING, ServerState.ACTIVE, ServerState.DRAINING]
 
_NORM_RESPONSE_TIME_SEC = 0.098    
                                   
_NORM_ENERGY_JOULE = 10_221.51     
                                   
_NORM_ARRIVAL_RATE = 3.0           
NORM_RESPONSE_TIME_SEC = _NORM_RESPONSE_TIME_SEC
NORM_ENERGY_JOULE = _NORM_ENERGY_JOULE
NORM_ARRIVAL_RATE = _NORM_ARRIVAL_RATE


def build_state_vector(snapshot: dict, servers: dict) -> np.ndarray:
    parts = []

    for sid in sorted(CFG.server_info.keys()):
        s_snap = snapshot["servers"][sid]
        one_hot = [1.0 if s_snap["state"] == st else 0.0 for st in _SERVER_STATE_ORDER]
        n_replicas = len(servers[sid].hosted_replicas)
        parts.extend(one_hot)
        parts.append(float(s_snap["utilization"]))
        parts.append(n_replicas / 15.0)  

    for svc_id in CFG.active_services:
        sv = snapshot["services"][svc_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
        parts.append(sv["n_replicas"] / CFG.n_servers)
        parts.append(min(occ_ratio, 2.0) / 2.0)
        parts.append(sv["deadline_violation_rate"])
        parts.append(min(sv["recent_arrivals"] / _NORM_ARRIVAL_RATE, 2.0) / 2.0) 
        parts.append(float(sv.get("rejection_rate", 0.0)))
        parts.append(float(sv.get("proximity_violation_rate", 0.0)))

    g = snapshot["global"]
    parts.append(min(g["avg_response_time_recent"] / _NORM_RESPONSE_TIME_SEC, 2.0) / 2.0)
    parts.append(min(g["energy_recent_joule"] / _NORM_ENERGY_JOULE, 2.0) / 2.0)

    vec = np.array(parts, dtype=np.float32)
    assert vec.shape[0] == STATE_DIM, f"state dim mismatch: {vec.shape[0]} != {STATE_DIM}"
    return vec