"""
Server Scaler (*** بازطراحی: مبتنی بر utilization، نه on-demand ***)

مسئولیت: روشن/خاموش کردن خودِ سرورهای فیزیکی، مستقل از اینکه هر سرویس چند
replica می‌خواهد.

سیاست:
    - شبیه‌سازی با `SERVER_INITIAL_COUNT` سرور روشن شروع می‌شود (نه همه‌شان).
    - هر چرخه، میانگین utilization سرورهای روشن بر اساس بار *چرخه‌ی قبل*
      محاسبه می‌شود: utilization = (مجموع بار همه‌ی سرورهای روشن) /
      (تعداد سرور روشن × SERVER_CAPACITY_PER_CYCLE).
    - اگر utilization > SERVER_SCALE_UP_UTIL: یک سرور خاموشِ دیگر روشن می‌شود
      (با تاخیر boot طبق SERVER_BOOT_DELAY_CYCLES).
    - اگر utilization < SERVER_SCALE_DOWN_UTIL: کم‌بارترین سرور روشن، خاموش
      می‌شود (تا حداقل SERVER_MIN_COUNT سرور روشن بماند).
    - در هر چرخه حداکثر یک تغییر (یک روشن یا یک خاموش) اعمال می‌شود تا رفتار
      نوسانی/jumpy نداشته باشیم.

نکته‌ی مهم درباره‌ی توالی زمانی: چون utilization فقط از بار *چرخه‌ی قبل*
قابل‌محاسبه است (بار همین چرخه هنوز مشخص نیست)، این یک کنترل واکنشی با یک
چرخه تاخیر است (دقیقاً مثل HPA واقعی کوبرنتیز). به همین دلیل متد اصلی این
کلاس دو بخش دارد: `ready_mask()` (قبل از dispatch این چرخه صدا زده می‌شود تا
service scalerها بدانند کجا مجازند replica بگذارند) و `update(...)` (بعد از
dispatch همین چرخه صدا زده می‌شود تا آماده‌ی تصمیم چرخه‌ی بعد شود).
"""

from __future__ import annotations
import numpy as np

from common.config import CFG


class ServerScaler:
    name = "ServerScaler"

    def __init__(self, n_servers: int,
                 initial_count: int = CFG.SERVER_INITIAL_COUNT,
                 min_count: int = CFG.SERVER_MIN_COUNT,
                 capacity_per_cycle: float = CFG.SERVER_CAPACITY_PER_CYCLE,
                 scale_up_util: float = CFG.SERVER_SCALE_UP_UTIL,
                 scale_down_util: float = CFG.SERVER_SCALE_DOWN_UTIL,
                 boot_delay: int = CFG.SERVER_BOOT_DELAY_CYCLES):
        self.n_servers = n_servers
        self.min_count = min_count
        self.capacity_per_cycle = capacity_per_cycle
        self.scale_up_util = scale_up_util
        self.scale_down_util = scale_down_util
        self.boot_delay = boot_delay

        self.on_mask = np.zeros(n_servers, dtype=bool)
        self.on_mask[:initial_count] = True          # *** شروع با فقط initial_count سرور
        self.booting_remaining = np.zeros(n_servers, dtype=int)

        self.total_power_on_events = 0
        self.total_power_off_events = 0
        self.last_utilization = 0.0

    def ready_mask(self) -> np.ndarray:
        """سرورهایی که *همین الان* روشن و از دوره‌ی boot رد شده‌اند (برای service scalerها)."""
        return self.on_mask & (self.booting_remaining <= 0)

    def update(self, load_per_server_last_cycle: np.ndarray):
        """
        بعد از dispatch هر چرخه صدا زده می‌شود. بر اساس بار *همین چرخه‌ای که
        الان تمام شد*، تصمیم روشن/خاموش کردن سرور برای چرخه‌ی بعد را می‌گیرد.
        """
        # کاهش شمارنده‌ی boot برای سرورهای در حال بالا آمدن
        booting = self.on_mask & (self.booting_remaining > 0)
        self.booting_remaining[booting] -= 1

        on_idx = np.where(self.on_mask)[0]
        n_on = len(on_idx)
        total_load = float(load_per_server_last_cycle[on_idx].sum()) if n_on else 0.0
        capacity_total = n_on * self.capacity_per_cycle
        utilization = (total_load / capacity_total) if capacity_total > 0 else 1.0
        self.last_utilization = utilization

        if utilization > self.scale_up_util:
            off_idx = np.where(~self.on_mask)[0]
            if len(off_idx) > 0:
                nxt = int(off_idx[0])  # *** ترتیب ثابت افزودن (سرورهای k-means از قبل پخش‌اند)
                self.on_mask[nxt] = True
                self.booting_remaining[nxt] = self.boot_delay
                self.total_power_on_events += 1

        elif utilization < self.scale_down_util and n_on > self.min_count:
            # کم‌بارترین سرور روشن را خاموش کن
            loads_on = load_per_server_last_cycle[on_idx]
            victim = int(on_idx[int(np.argmin(loads_on))])
            self.on_mask[victim] = False
            self.booting_remaining[victim] = 0
            self.total_power_off_events += 1
