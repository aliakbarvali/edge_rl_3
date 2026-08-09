"""
common/network_coordinates.py

"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np


class VivaldiCoordinate:

    DIM = 2

    def __init__(self, rng: np.random.Generator):
        self.vec = rng.uniform(-1.0, 1.0, size=self.DIM) * 0.01
        self.height = float(rng.uniform(0.05, 0.3))
        self.error_estimate = 2.0

    def predicted_rtt_ms(self, other: "VivaldiCoordinate") -> float:
        euclid = float(np.linalg.norm(self.vec - other.vec))
        return euclid + self.height + other.height

    def update(self, other: "VivaldiCoordinate", observed_rtt_ms: float,
               rng: np.random.Generator, ce: float = 0.25, cc: float = 0.5) -> None:
        predicted = self.predicted_rtt_ms(other)
        error = abs(predicted - observed_rtt_ms)
        w = self.error_estimate / (self.error_estimate + other.error_estimate + 1e-9)

        rel_error = min(error / max(observed_rtt_ms, 1e-6), 1.0)
        alpha = ce * w
        self.error_estimate = alpha * rel_error * self.error_estimate + (1 - alpha) * self.error_estimate
        self.error_estimate = max(self.error_estimate, 0.05)  
        delta = cc * w
        direction = self.vec - other.vec
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            unit = rng.uniform(-1.0, 1.0, size=self.DIM)
            unit = unit / (float(np.linalg.norm(unit)) + 1e-9)
        else:
            unit = direction / norm
        self.vec = self.vec + delta * (observed_rtt_ms - predicted) * unit


class VivaldiNetwork:
    def __init__(self, servers: dict, base_latency_ms: float, k_ms_per_km: float,
                 seed: int = 0, bootstrap_rounds: int = 20):
        from common.geo import haversine_km, network_delay_ms 

        self._rng = np.random.default_rng(seed)
        self._server_coords: Dict[int, VivaldiCoordinate] = {
            sid: VivaldiCoordinate(self._rng) for sid in servers
        }
        self._bts_coords: Dict[Tuple[float, float], VivaldiCoordinate] = {}
        self._bts_observations: Dict[Tuple[float, float], int] = defaultdict(int)

        ids = list(servers.keys())
        for _ in range(bootstrap_rounds):
            for i in ids:
                for j in ids:
                    if i == j:
                        continue
                    true_rtt = 2 * network_delay_ms(
                        haversine_km(servers[i].lat, servers[i].long, servers[j].lat, servers[j].long),
                        base_latency_ms, k_ms_per_km)
                    self._server_coords[i].update(self._server_coords[j], true_rtt, self._rng)

    @staticmethod
    def _bts_key(lat: float, lon: float) -> Tuple[float, float]:
        return (round(lat, 5), round(lon, 5))

    def _get_or_create_bts_coord(self, lat: float, lon: float) -> VivaldiCoordinate:
        key = self._bts_key(lat, lon)
        coord = self._bts_coords.get(key)
        if coord is None:
            coord = VivaldiCoordinate(self._rng)
            self._bts_coords[key] = coord
        return coord

    def estimate_rtt_ms(self, bts_lat: float, bts_lon: float, server_id: int) -> float:
        
        bts_coord = self._get_or_create_bts_coord(bts_lat, bts_lon)
        return bts_coord.predicted_rtt_ms(self._server_coords[server_id])

    def observe(self, bts_lat: float, bts_lon: float, server_id: int, true_rtt_ms: float) -> None:
        key = self._bts_key(bts_lat, bts_lon)
        bts_coord = self._get_or_create_bts_coord(bts_lat, bts_lon)
        bts_coord.update(self._server_coords[server_id], true_rtt_ms, self._rng)
        self._bts_observations[key] += 1

    def observation_count(self, bts_lat: float, bts_lon: float) -> int:
        return self._bts_observations.get(self._bts_key(bts_lat, bts_lon), 0)