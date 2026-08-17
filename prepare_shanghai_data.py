"""
prepare_shanghai_data.py

پالایش یک‌باره‌ی داده‌ی خام شانگهای به فرمت مورد انتظار پروژه
(id,BTSID,Lat,Long,ServiceID,startSec) طبق data/loader.py:COLUMNS.

این اسکریپت یک پیش‌پردازش آفلاین است (نه بخشی از loader.py که هر بار در
حین اجرای شبیه‌سازی صدا زده می‌شود) - چون حجم داده‌ی خام خیلی زیاد است و
منطق پالایش (تشخیص آرتیفکت جمع‌آوری، فیلتر جغرافیایی بر پایه‌ی فاصله از
سرورها) گران و یک‌بار-مصرف است، نه چیزی که باید هر اجرای run.py دوباره
حساب شود. خروجی این اسکریپت فایل‌های تمیز DataN.csv است که مستقیم در
data/raw/ می‌روند و توسط data/loader.py فعلی (بدون هیچ تغییری) خوانده
می‌شوند.

سه پالایش اصلی (طبق تشخیص شما + اعداد واقعی‌ای که روی نمونه‌تان اندازه
گرفتیم):

  ۱) فیلتر جغرافیایی بر پایه‌ی فاصله از سرورها (نه باکس کل شانگهای):
     فقط رویدادهایی که در شعاع RADIUS_KM (پیش‌فرض ۱۲) از حداقل یکی از
     ۱۰ سرور واقعی پروژه (common/config.py:SERVER_INFO) هستند نگه داشته
     می‌شوند. روی نمونه‌ی شما ۸۴.۱٪ باقی می‌ماند.

  ۲) رفع آرتیفکت جمع‌آوری در دقیقه‌ی صفر بعضی ساعت‌ها:
     تشخیص داده شد که فقط ساعت‌های ۰۷ و ۲۳ در این نمونه دچار این مشکل‌اند
     (به ترتیب ۱۸.۴× و ۲۷.۷× پرتر از میانه‌ی بقیه‌ی دقایق همان ساعت)؛ بقیه‌ی
     ۲۲ ساعت کاملاً طبیعی‌اند (نسبت ۰.۸ تا ۱.۵×). این اسکریپت به‌جای فیلتر
     دستی/هاردکدشده‌ی «فقط ۷ و ۲۳»، آن را *به‌صورت آماری* برای هر (روز,
     ساعت) تشخیص می‌دهد (SPIKE_RATIO_THRESHOLD) تا اگر در کل دیتای شما
     الگوی دیگری هم بود، خودکار پیدا شود - این برای دفاع هم مهم است: معیار
     تشخیص عینی و قابل‌تکرار است، نه چشمی/دستی.
     برای هر (روز, ساعت) آرتیفکت‌دار: فقط به‌اندازه‌ی میانه‌ی بقیه‌ی دقایق آن
     ساعت در دقیقه‌ی صفر نگه داشته می‌شود؛ مازاد به‌طور یکنواخت در بازه‌ی
     ±REDISTRIBUTE_WINDOW_MIN دقیقه (پیش‌فرض ۱۵) پخش می‌شود - یعنی حجم کل
     روز حفظ می‌شود، فقط شکل غیرواقعی‌اش اصلاح می‌شود.

  ۳) تزریق ثانیه:
     چون timestamp خام فقط دقیقه دارد (ثانیه همیشه ۰ است)، برای همه‌ی
     رکوردها (چه اصلاح‌شده چه نشده) jitter یکنواخت ۰-۵۹ ثانیه اضافه می‌شود.

اجرا:
    python3 prepare_shanghai_data.py --input DataAll.csv --output-dir data/raw \
        --radius-km 12 --spike-ratio 5 --redistribute-window-min 15
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

# --- ۱۰ سرور پروژه (common/config.py:SERVER_INFO) - برای فیلتر فاصله ---
SERVER_LOCATIONS = {
    1: (31.37, 121.25), 2: (31.31, 121.51), 3: (31.10, 121.18), 4: (31.25, 121.37),
    5: (31.10, 121.36), 6: (31.04, 121.74), 7: (31.17, 121.57), 8: (31.15, 121.41),
    9: (31.20, 121.43), 10: (31.16, 121.49),
}

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_raw(path: str) -> pd.DataFrame:
    """هر دو فرمتی که تا حالا دیده‌ایم (نسخه‌ی کامل با id/user_id/... و نسخه‌ی
    خام‌تر DataAll.csv بدون آن ستون‌ها) را می‌پذیرد."""
    df = pd.read_csv(path)
    df["StartTime"] = pd.to_datetime(df["StartTime"])
    return df


def filter_by_server_radius(df: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    min_dist = np.full(len(df), np.inf)
    for _sid, (slat, slon) in SERVER_LOCATIONS.items():
        d = haversine_km(df.Lat.values, df.Long.values, slat, slon)
        min_dist = np.minimum(min_dist, d)
    kept = df[min_dist <= radius_km].copy()
    print(f"  فیلتر شعاع {radius_km}km از سرورها: {len(kept)}/{len(df)} "
          f"({100 * len(kept) / max(len(df), 1):.1f}%) نگه داشته شد")
    return kept


def fix_collection_spikes(df: pd.DataFrame, spike_ratio: float,
                           redistribute_window_min: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = df.StartTime.dt.date
    df["_hour"] = df.StartTime.dt.hour
    df["_minute"] = df.StartTime.dt.minute

    fixed_parts = []
    n_spike_groups = 0
    for (date, hour), g in df.groupby(["_date", "_hour"]):
        zero_min = g[g._minute == 0]
        rest = g[g._minute != 0]
        rest_counts_per_min = rest.groupby("_minute").size()
        baseline = rest_counts_per_min.median() if len(rest_counts_per_min) else 0.0
        ratio = len(zero_min) / max(baseline, 1.0)

        if len(zero_min) > 5 and ratio > spike_ratio:
            n_spike_groups += 1
            keep_n = int(round(baseline))
            keep_idx = rng.choice(zero_min.index, size=min(keep_n, len(zero_min)), replace=False)
            keep_rows = zero_min.loc[keep_idx]
            excess_rows = zero_min.drop(index=keep_idx).copy()
            jitter_min = rng.uniform(-redistribute_window_min, redistribute_window_min, size=len(excess_rows))
            excess_rows["StartTime"] = excess_rows["StartTime"] + pd.to_timedelta(jitter_min, unit="m")
            fixed_parts.extend([keep_rows, excess_rows, rest])
        else:
            fixed_parts.append(g)

    print(f"  آرتیفکت جمع‌آوری: {n_spike_groups} گروه (روز,ساعت) با نسبت > {spike_ratio}x تشخیص و اصلاح شد")
    out = pd.concat(fixed_parts, ignore_index=True)
    return out.drop(columns=["_date", "_hour", "_minute"])


def inject_second_jitter(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    secs = rng.integers(0, 60, size=len(df))
    df["StartTime"] = df["StartTime"] + pd.to_timedelta(secs, unit="s")
    return df


def to_project_format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # BTSID ممکن است هش رشته‌ای باشد (DataAll.csv) یا از قبل عددی -
    # factorize در هر دو حالت یک شناسه‌ی عددی پایدار درون همان روز می‌سازد.
    out["BTSID"] = pd.factorize(out["BTSID"])[0] + 1
    day_start = out["StartTime"].dt.normalize()
    out["startSec"] = (out["StartTime"] - day_start).dt.total_seconds().astype(int)
    out = out.sort_values("startSec").reset_index(drop=True)
    out["id"] = np.arange(1, len(out) + 1)
    return out[["id", "BTSID", "Lat", "Long", "ServiceID", "startSec"]]


def process_one_day(df_day: pd.DataFrame, radius_km: float, spike_ratio: float,
                     redistribute_window_min: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_day = filter_by_server_radius(df_day, radius_km)
    df_day = fix_collection_spikes(df_day, spike_ratio, redistribute_window_min, rng)
    df_day = inject_second_jitter(df_day, rng)
    return to_project_format(df_day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="مسیر CSV خام (یک یا چند روز)")
    ap.add_argument("--output-dir", default="data/raw")
    ap.add_argument("--radius-km", type=float, default=12.0)
    ap.add_argument("--spike-ratio", type=float, default=5.0)
    ap.add_argument("--redistribute-window-min", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="Day")
    args = ap.parse_args()

    raw = load_raw(args.input)
    print(f"ورودی: {len(raw)} رکورد خام")

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    for i, date in enumerate(sorted(raw.StartTime.dt.date.unique()), start=1):
        day_df = raw[raw.StartTime.dt.date == date]
        weekday = day_df.StartTime.dt.day_name().iloc[0]
        span_hours = (day_df.StartTime.max() - day_df.StartTime.min()).total_seconds() / 3600
        print(f"\n--- {date} ({weekday}), {len(day_df)} رکورد خام، بازه {span_hours:.1f}h ---")
        if span_hours < 20:
            print(f"  !! هشدار: این روز کمتر از ۲۰ ساعت پوشش دارد (احتمالاً ناقص است) - "
                  f"برای فایل نهایی پیشنهاد می‌شود کنار گذاشته شود مگر عمداً بخواهید.")
        clean = process_one_day(day_df, args.radius_km, args.spike_ratio,
                                 args.redistribute_window_min, args.seed + i)
        out_path = os.path.join(args.output_dir, f"{args.prefix}_{date}.csv")
        clean.to_csv(out_path, index=False)
        print(f"  -> ذخیره شد: {out_path} ({len(clean)} رکورد نهایی)")


if __name__ == "__main__":
    main()
