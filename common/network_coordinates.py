"""
common/network_coordinates.py

پیاده‌سازی سبک Vivaldi (Dabek, Cox, Kaashoek, Morris - "Vivaldi: A Decentralized
Network Coordinate System", SIGCOMM 2004) - دقیقاً همان سیستمی که مقاله‌ی VOILA
برای تخمین RTT واقعی (بدون دانش پیشینی از توپولوژی/موقعیت جغرافیایی) استفاده
می‌کند و در بخش ۵ سند معماری این پروژه به آن ارجاع داده شده.

*** تفاوت بنیادی با حالت فعلی پروژه (haversine مستقیم):
در حالت فعلی، انتخاب رپلیکا بر پایه‌ی فاصله‌ی جغرافیایی *واقعی* (oracle) است -
یعنی سیستم از قبل دقیقاً می‌داند هر سرور کجاست. این ماژول این فرض را می‌شکند:
هر نود (سرور یا BTS) فقط یک بردار مختصات مجازی دارد که ابتدا تصادفی/نزدیک صفر
است و *فقط* از طریق مشاهده‌ی RTT واقعیِ تعامل‌های قبلی (نه دانش پیشینی)
به‌تدریج به یک تخمین قابل‌قبول از RTT واقعی همگرا می‌شود - دقیقاً مثل یک
Vivaldi واقعی. نتیجه: در ابتدای اجرا (cold start) تخمین‌ها ناقص‌اند و تصمیمات
روتینگ ممکن است غیربهینه باشند؛ با گذشت زمان و انباشت مشاهدات دقیق‌تر می‌شوند.

*** ساده‌سازی‌های آگاهانه نسبت به مقاله‌ی اصلی (مستند، نه پنهان):
1. مدل height-vector کامل مقاله (که خودش هم آپدیت می‌شود) به یک مقدار height
   ثابت تصادفی per-node ساده شده - نماینده‌ی «تاخیر لینک دسترسی محلی».
2. سرورها (زیرساخت ثابت و از پیش شناخته‌شده - برخلاف BTSهای مشتری که کاملاً
   ناشناخته‌اند) از طریق چند دور "landmark bootstrap" (پینگ متقابل واقعی بین
   خودشان، قبل از شروع شبیه‌سازی) از قبل کالیبره می‌شوند - این دقیقاً معادل
   تکنیک رایج landmark-based bootstrapping در سیستم‌های Vivaldi واقعی
   (مثلاً Azureus/Vuze) است، نه تقلبی برای دور زدن مسئله.
3. مختصات هر BTS فقط بعد از رسیدن *اولین* درخواست از آن BTS ساخته می‌شود
   (lazy) و فقط از طریق مشاهده‌ی RTT واقعیِ رپلیکایی که واقعاً به آن روت شده
   به‌روزرسانی می‌شود - نه از هر ۱۰ سرور، فقط همانی که انتخاب شده (مشابه یک
   client واقعی که فقط با peerهایی که واقعاً ارتباط برقرار کرده RTT دارد).
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np


class VivaldiCoordinate:
    """یک نود در فضای مختصات مجازی (بردار ۲بعدی + height ثابت)."""

    DIM = 2

    def __init__(self, rng: np.random.Generator):
        # شروع نزدیک مبدأ با کمی نویز تصادفی (نه صفر مطلق - برای جلوگیری از
        # تقارن کامل که باعث می‌شود جهت اولین آپدیت نامعین باشد)
        self.vec = rng.uniform(-1.0, 1.0, size=self.DIM) * 0.01
        self.height = float(rng.uniform(0.05, 0.3))
        # error_estimate بالا در ابتدا = "می‌دانم که نمی‌دانم"؛ با آپدیت‌های
        # موفق (خطای کم بین پیش‌بینی و RTT واقعی) کاهش می‌یابد.
        self.error_estimate = 2.0

    def predicted_rtt_ms(self, other: "VivaldiCoordinate") -> float:
        euclid = float(np.linalg.norm(self.vec - other.vec))
        return euclid + self.height + other.height

    def update(self, other: "VivaldiCoordinate", observed_rtt_ms: float,
               rng: np.random.Generator, ce: float = 0.25, cc: float = 0.5) -> None:
        """قانون spring-relaxation مقاله (بخش ۳.۴ اصلی، ساده‌شده)."""
        predicted = self.predicted_rtt_ms(other)
        error = abs(predicted - observed_rtt_ms)
        w = self.error_estimate / (self.error_estimate + other.error_estimate + 1e-9)

        rel_error = min(error / max(observed_rtt_ms, 1e-6), 1.0)
        # *** رفع باگ (بازبینی): طبق pseudocode اصلی مقاله‌ی Vivaldi
        # (Dabek et al. 2004, Figure 3)، وزن بروزرسانی error_estimate باید
        # ce*w باشد و وزن بروزرسانی مختصات (delta) باید cc*w باشد. اینجا
        # قبلاً این دو کاملاً جابجا بودند (alpha از cc و delta از ce
        # استفاده می‌کرد) - با مقادیر پیش‌فرض (ce=0.25, cc=0.5) یعنی
        # error_estimate سریع‌تر از حد انتظار decay می‌کرد و مختصات
        # کندتر از حد انتظار حرکت می‌کردند، که همگرایی RTT تخمینی Voila را
        # کندتر/نادرست‌تر از طراحی اصلی می‌کرد.
        alpha = ce * w
        self.error_estimate = alpha * rel_error * self.error_estimate + (1 - alpha) * self.error_estimate
        self.error_estimate = max(self.error_estimate, 0.05)  # کف - از overconfidence کاذب جلوگیری می‌کند

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
    """
    مدیریت مختصات همه‌ی نودها (سرورها + BTSهای دیده‌شده) برای یک اجرای واحد
    شبیه‌سازی. با servers ساخته می‌شود (چون بوت‌استرپ سرورها به موقعیت واقعی‌شان
    نیاز دارد - نگاه کنید داکیومنت بالای فایل، بند ۲).
    """

    def __init__(self, servers: dict, base_latency_ms: float, k_ms_per_km: float,
                 seed: int = 0, bootstrap_rounds: int = 20):
        from common.geo import haversine_km, network_delay_ms  # جلوگیری از import چرخه‌ای

        self._rng = np.random.default_rng(seed)
        self._server_coords: Dict[int, VivaldiCoordinate] = {
            sid: VivaldiCoordinate(self._rng) for sid in servers
        }
        self._bts_coords: Dict[Tuple[float, float], VivaldiCoordinate] = {}
        self._bts_observations: Dict[Tuple[float, float], int] = defaultdict(int)

        # *** بوت‌استرپ landmark: سرورها زیرساخت ثابت/شناخته‌شده‌اند، پس اجازه
        # داریم چند دور پینگ متقابل واقعی بین خودشان انجام دهیم تا مختصاتشان
        # قبل از ورود اولین درخواست همگرا شده باشد.
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
        """تخمین *فعلی* (ممکن است هنوز ناقص باشد) RTT، نه مقدار واقعی."""
        bts_coord = self._get_or_create_bts_coord(bts_lat, bts_lon)
        return bts_coord.predicted_rtt_ms(self._server_coords[server_id])

    def observe(self, bts_lat: float, bts_lon: float, server_id: int, true_rtt_ms: float) -> None:
        """بعد از هر درخواست واقعاً روت‌شده، این یک 'پینگ واقعی' حساب می‌شود
        و مختصات BTS مربوطه را کمی به سمت واقعیت اصلاح می‌کند."""
        key = self._bts_key(bts_lat, bts_lon)
        bts_coord = self._get_or_create_bts_coord(bts_lat, bts_lon)
        bts_coord.update(self._server_coords[server_id], true_rtt_ms, self._rng)
        self._bts_observations[key] += 1

    def observation_count(self, bts_lat: float, bts_lon: float) -> int:
        return self._bts_observations.get(self._bts_key(bts_lat, bts_lon), 0)