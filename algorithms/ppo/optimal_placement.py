# algorithms/ppo/optimal_placement.py
"""
جای‌گذاری اولیه‌ی *چندهدفه* برای PPO: به‌جای Set-Cover خالص (که فقط تعداد
سرور را کمینه می‌کند و کاملاً کور به انرژی/کیفیت مسافت است)، یک مدل ترکیبی
Set-Cover + p-median حل می‌شود که هم‌زمان سه هدف را با وزن قابل‌تنظیم بهینه
می‌کند:
    ۱) تعداد سرور فعال (هزینه‌ی ثابت هر سرور)
    ۲) مجموع توان idle سرورهای انتخاب‌شده (p_idle هر پروفایل متفاوت است -
       large=110W خیلی گران‌تر از edge_small=40W حتی در حالت بیکار)
    ۳) کیفیت مسافت (میانگین وزن‌دار فاصله‌ی هر BTS تا نزدیک‌ترین سرور
       *انتخاب‌شده*ی واقعی که به آن تخصیص یافته - نه صرفاً «زیر آستانه»)

*** چرا p-median و نه صرفاً Set-Cover: در Set-Cover خالص، سؤال فقط این است
«آیا این BTS توسط حداقل یک سرور پوشش داده می‌شود؟» (باینری). این باعث می‌شود
solver با کمترین تعداد سرور ممکن (حتی فقط ۱-۲ سرور بزرگ) به جواب "بهینه"
برسد، چون آستانه‌ی پوشش (l0_ms) نسبت به ابعاد واقعی شهر بسیار سخاوتمندانه
است (~900km). اما این یعنی BTSهای دورافتاده هم به همان سرور دور تخصیص
می‌یابند و کیفیت مسیریابی واقعی (بخش ۳ سند: network_delay وابسته به فاصله‌ی
واقعی) فاجعه‌بار می‌شود. با افزودن متغیر تخصیص x[i,j] و جریمه‌ی فاصله در
هدف، solver دیگر انگیزه‌ای برای «کمینه‌سازی صرف تعداد به هر قیمت» ندارد.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from common.config import CFG
from common.geo import haversine_km


def aggregate_training_demand(train_events_df) -> List[Tuple[float, float, int]]:
    """(lat, lon, weight) برای هر BTS یکتا در کل تایم‌لاین train، weight = تعداد درخواست."""
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
    """
    مدل ترکیبی Set-Cover + p-median با ۳ جزء هدف نرمال‌شده (وزن‌های
    w_count/w_energy/w_distance قابل کالیبراسیون - بخش ۱۳ سند، مشابه
    PPO_REWARD_WEIGHTS). خروجی: [] اگر solver به جواب feasible نرسید
    (فراخواننده باید fallback کند).
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError("کتابخانه‌ی pulp نصب نیست. اجرا کنید: pip install pulp") from e

    l0_ms = l0_ms if l0_ms is not None else CFG.l0_ms
    if min_total_capacity is None:
        min_total_capacity = sum(s["cpu_demand"] for s in CFG.services_info.values())
    server_ids = list(servers.keys())
    n_points = len(demand_points)

    # coverage[sid] = مجموعه‌ی اندیس نقاط تقاضایی که این سرور می‌تواند پوشش دهد
    # (حداکثر فاصله‌ی قابل قبول - قید سخت، نه بخشی از هدف نرم)
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
    # x[i, sid]: BTS شماره‌ی i به سرور sid تخصیص یافته (فقط برای جفت‌های
    # قابل‌پوشش تعریف می‌شود - کاهش تعداد متغیرها)
    x = {}
    for idx in coverable:
        for sid in server_ids:
            if idx in coverage[sid]:
                x[(idx, sid)] = pulp.LpVariable(f"x_{idx}_{sid}", cat="Binary")

    # --- قید ۱ (Set-Cover کلاسیک): هر BTS قابل‌پوشش دقیقاً به یک سرور تخصیص یابد
    for idx in coverable:
        prob += pulp.lpSum(x[(idx, sid)] for sid in server_ids if (idx, sid) in x) == 1

    # --- قید ۲: تخصیص فقط به سرور *انتخاب‌شده* مجاز است
    for (idx, sid) in x:
        prob += x[(idx, sid)] <= y[sid]

    # --- قید ۳: مجموع ظرفیت کافی برای هر ۱۵ سرویس
    prob += pulp.lpSum(servers[sid].capacity * y[sid] for sid in server_ids) >= min_total_capacity

    # ------------------------------------------------------------------
    # هدف نرمال‌شده: هر جزء بین تقریباً [0,1] مقیاس می‌شود تا هیچ‌کدام با
    # وزن پیش‌فرض ۱٫۰ بر بقیه غالب نشود (دقیقاً همان اصل نرمال‌سازی
    # PPO_REWARD_WEIGHTS در common/config.py).
    # ------------------------------------------------------------------
    n_servers_total = len(server_ids)
    total_p_idle = sum(_p_idle_of(servers, sid) for sid in server_ids)
    # مقیاس مرجع مسافت: بدترین حالت ممکن = هر BTS به دورترین سرور مجازش
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
    if status not in ("Optimal", "Not Solved"):  # Not Solved معمولاً یعنی timeout با یک جواب feasible موجود
        if status != "Optimal":
            return []

    selected = [sid for sid in server_ids if pulp.value(y[sid]) is not None and pulp.value(y[sid]) > 0.5]
    return selected if selected else []


def _p_idle_of(servers: Dict, sid: int) -> float:
    return servers[sid].p_idle