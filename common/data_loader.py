"""
بارگذاری Data4.csv و آماده‌سازی آن برای شبیه‌سازی چند-سرویسی event-level.

*** تغییر معماری مهم: گیت‌وی و سرور دیگر یکی نیستند:
    - gateways : همه‌ی BTSهای منحصربه‌فرد داخل محدوده‌ی جغرافیایی (بدون سقف) -
                 مبدأ واقعی درخواست‌ها؛ تا هیچ درخواستی به‌خاطر انتخاب زیرمجموعه
                 حذف نشود، این‌جا سقفی روی تعدادشان نمی‌گذاریم.
    - servers  : فقط CFG.N_SERVERS مکان کاندید میزبانی replica (مثلاً ۱۰، طبق
                 مقاله‌ی خودتان). هر گیت‌وی به نزدیک‌ترین سرور (از نظر latency)
                 route می‌شود - نگاه کنید به common/latency.py و dispatcher/.

قبلاً این دو یکی فرض می‌شدند (هر دو از میان پرترافیک‌ترین BTSها انتخاب
می‌شدند)؛ این باعث می‌شد وقتی تعداد سرور را کم می‌کردید (مثلاً برای تطبیق با
مقاله‌ی ۱۰-سروره‌ی خودتان)، بخش بزرگی از درخواست‌های واقعی (که مبدأشان جزو آن
چند سرور کم‌تعداد نبود) کلاً از شبیه‌سازی حذف شوند و total_requests کاذب پایین
بیاید.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from common.config import CFG


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CFG.DATA_PATH)
    df = df[(df.Lat.between(CFG.LAT_MIN, CFG.LAT_MAX)) &
            (df.Long.between(CFG.LON_MIN, CFG.LON_MAX))]
    return df[df.ServiceID.isin(CFG.ACTIVE_SERVICES)]


def select_gateways(df: pd.DataFrame) -> pd.DataFrame:
    """*** همه‌ی BTSهای منحصربه‌فرد منطقه (بدون سقف) به عنوان گیت‌وی."""
    gws = (df.groupby("BTSID")[["Lat", "Long"]].first()
           .reset_index().sort_values("BTSID").reset_index(drop=True))
    gws["gateway_idx"] = np.arange(len(gws))
    return gws


def select_servers(df: pd.DataFrame) -> pd.DataFrame:
    """
    *** انتخاب N_SERVERS مکان کاندید سرور با پخش جغرافیایی واقعی، نه صرفاً
    پرترافیک‌ترین BTSهای کل منطقه (که معمولاً همه در یک ناحیه‌ی شلوغ خوشه
    می‌شوند). روش: BTSها را با k-means (وزن‌دار بر اساس ترافیک) به N_SERVERS
    خوشه‌ی جغرافیایی تقسیم می‌کنیم؛ از هر خوشه پرترافیک‌ترین BTS واقعی (نه
    مرکز خوشه‌ی انتزاعی) به‌عنوان سرور انتخاب می‌شود - هم پخش جغرافیایی حفظ
    می‌شود هم سرور در یک مکان واقعاً موجود قرار می‌گیرد.
    """
    from sklearn.cluster import KMeans

    counts = df.groupby("BTSID").size().rename("count")
    coords = df.groupby("BTSID")[["Lat", "Long"]].first().join(counts)

    km = KMeans(n_clusters=CFG.N_SERVERS, random_state=CFG.SEED, n_init=10)
    coords["cluster"] = km.fit_predict(coords[["Lat", "Long"]], sample_weight=coords["count"])

    servers = (coords.loc[coords.groupby("cluster")["count"].idxmax()]
               .reset_index().sort_values("BTSID").reset_index(drop=True))
    servers["server_idx"] = np.arange(len(servers))
    return servers[["BTSID", "Lat", "Long", "server_idx"]]


def build_event_stream(df: pd.DataFrame, gateways: pd.DataFrame) -> pd.DataFrame:
    """
    جریان درخواست‌های تک‌تک (event-level) برای dispatcher.
    *** حالا همه‌ی درخواست‌های داخل محدوده‌ی جغرافیایی نگه داشته می‌شوند (چون
    گیت‌وی دیگر به سرورهای کم‌تعداد محدود نیست)، نه فقط آن‌هایی که مبدأشان جزو
    سرورهای انتخاب‌شده بود.
    خروجی: DataFrame مرتب‌شده بر اساس زمان با ستون‌های
        startSec, cycle, ServiceID, gateway_idx
    """
    events = df.copy()
    events["cycle"] = (events.startSec // CFG.TAU).clip(upper=CFG.N_CYCLES - 1)
    bts_to_idx = dict(zip(gateways.BTSID, gateways.gateway_idx))
    events["gateway_idx"] = events.BTSID.map(bts_to_idx)
    events = events[["startSec", "cycle", "ServiceID", "gateway_idx"]].sort_values("startSec")
    return events.reset_index(drop=True)


def cycle_loads_for_service(events: pd.DataFrame, service_id: int, n_gateways: int) -> np.ndarray:
    """
    آرایه‌ی (N_CYCLES, n_gateways) بار هر گیت‌وی در هر چرخه، فقط برای یک سرویس؛
    این آرایه توسط تخمین‌گر سریع E% داخل هر service scaler استفاده می‌شود
    (نه توسط dispatcher که مستقیماً از events استفاده می‌کند).
    """
    svc = events[events.ServiceID == service_id]
    loads = np.zeros((CFG.N_CYCLES, n_gateways), dtype=np.float64)
    grouped = svc.groupby(["cycle", "gateway_idx"]).size()
    for (c, i), cnt in grouped.items():
        loads[int(c), int(i)] = cnt
    return loads


def build_service_profile(df: pd.DataFrame) -> dict:
    """{service_id: {"execution_time_sec": ..., "resource": ...}} - مقادیر
    ثابت هر سرویس در Data4.csv؛ برای response time/deadline در dispatcher."""
    prof = df.groupby("ServiceID")[["ServiceResource", "ServiceExecutionTime"]].first()
    return {
        int(sid): {"execution_time_sec": float(row.ServiceExecutionTime),
                    "resource": float(row.ServiceResource)}
        for sid, row in prof.iterrows()
    }


def prepare_dataset():
    """نقطه‌ی ورودی اصلی: خروجی (gateways_df, servers_df, event_stream_df, service_profile_dict)."""
    df = load_raw()
    gateways = select_gateways(df)
    servers = select_servers(df)
    events = build_event_stream(df, gateways)
    profile = build_service_profile(df)
    return gateways, servers, events, profile


if __name__ == "__main__":
    gateways, servers, events, profile = prepare_dataset()
    print(f"تعداد گیت‌وی (همه BTSهای منطقه): {len(gateways)}")
    print(f"تعداد سرور کاندید: {len(servers)}")
    print(f"تعداد کل رویدادهای درخواست (همه‌ی {len(CFG.ACTIVE_SERVICES)} سرویس): {len(events)}")
    print(f"میانگین رویداد در هر چرخه: {len(events) / CFG.N_CYCLES:.1f}")
    per_service = events.groupby("ServiceID").size().sort_values(ascending=False)
    print("توزیع رویداد بر اساس سرویس:")
    print(per_service)
    print("پروفایل زمان اجرای هر سرویس (ثانیه):", profile)
