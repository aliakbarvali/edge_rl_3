"""
common/geo.py
توابع فاصله‌ی جغرافیایی (هاورسین) و مدل تاخیر شبکه، طبق بخش ۳ سند معماری.

network_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM * distance_km
"""

from __future__ import annotations
import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """فاصله‌ی هاورسین بین دو نقطه (کیلومتر)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_KM * c


def network_delay_ms(distance_km: float, base_latency_ms: float, k_ms_per_km: float) -> float:
    return base_latency_ms + k_ms_per_km * distance_km
