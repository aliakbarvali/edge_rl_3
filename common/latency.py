"""
ساخت ماتریس تاخیر L (مطابق Table I مقاله: L[l_ij]).

*** تغییر مهم: از این نسخه به بعد L مستطیلی است، نه مربعی: سطرها = گیت‌وی‌ها
(همه‌ی BTSهای منطقه، مبدأ درخواست‌ها) و ستون‌ها = سرورها (فقط N_SERVERS مکان
کاندید میزبانی replica، طبق مقاله‌ی خودتان مثلاً ۱۰ سرور). این جدایی لازم بود
چون قبلاً گیت‌وی و سرور یکی فرض می‌شدند و کاهش تعداد سرور باعث می‌شد بخش
بزرگی از درخواست‌های واقعی (که مبدأشان جزو سرورهای کم‌تعداد نبود) کلاً حذف شوند.

مقاله تاخیر واقعی را با دستور Linux tc به صورت "تابعی خطی از فاصله‌ی جغرافیایی"
شبیه‌سازی کرده و بازه‌ی ۴ تا ۸۰ میلی‌ثانیه با میانه‌ی ۲۶ میلی‌ثانیه به‌دست آورده
(بخش V-A). همین رویکرد را اینجا پیاده می‌کنیم.
"""

from __future__ import annotations
import numpy as np

from common.config import CFG

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray,
                      lat2: np.ndarray | None = None, lon2: np.ndarray | None = None) -> np.ndarray:
    """
    ماتریس فاصله (km) با فرمول هاورسین.
    اگر فقط (lat1, lon1) داده شود: ماتریس مربعی فاصله‌ی هر جفت از همان مجموعه.
    اگر (lat2, lon2) هم داده شود: ماتریس مستطیلی fasele بین مجموعه‌ی ۱ (سطرها)
    و مجموعه‌ی ۲ (ستون‌ها) - مثلاً گیت‌وی‌ها × سرورها.
    """
    if lat2 is None:
        lat2, lon2 = lat1, lon1
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat1_r[:, None] - lat2_r[None, :]
    dlon = lon1_r[:, None] - lon2_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1_r[:, None]) * np.cos(lat2_r[None, :]) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def build_latency_matrix(gw_lat: np.ndarray, gw_lon: np.ndarray,
                          srv_lat: np.ndarray | None = None, srv_lon: np.ndarray | None = None,
                          lat_min_ms: float = CFG.LATENCY_MIN_MS,
                          lat_max_ms: float = CFG.LATENCY_MAX_MS) -> np.ndarray:
    """
    فاصله -> تاخیر (ms) با نگاشت خطی min-max؛ خروجی شکل (n_gateways, n_servers).
    اگر srv_lat/srv_lon داده نشود، مربعی (سازگاری با کد قدیمی) برمی‌گردد.
    """
    dist_km = haversine_matrix(gw_lat, gw_lon, srv_lat, srv_lon)
    max_dist = dist_km.max()
    if max_dist == 0:
        return np.zeros(dist_km.shape)
    L = lat_min_ms + (dist_km / max_dist) * (lat_max_ms - lat_min_ms)
    L[dist_km == 0] = 0.0  # فاصله‌ی صفر (سرور دقیقاً روی همان مختصات گیت‌وی) -> تاخیر صفر
    return L


if __name__ == "__main__":
    from common.data_loader import prepare_dataset
    gateways, servers, _, _ = prepare_dataset()
    L = build_latency_matrix(gateways.Lat.values, gateways.Long.values,
                              servers.Lat.values, servers.Long.values)
    print(f"شکل L (گیت‌وی×سرور): {L.shape}")
    print(f"min={L.min():.1f}ms median={np.median(L):.1f}ms max={L.max():.1f}ms")
