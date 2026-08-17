"""
data/loader.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from common.config import CFG

COLUMNS = ["id", "BTSID", "Lat", "Long", "ServiceID", "startSec"]


def _load_one_day(filename: str, day_index: int) -> pd.DataFrame:
    path = os.path.join(CFG.data_dir, filename)
    df = pd.read_csv(path, usecols=COLUMNS)
    df = df[(df.Lat.between(CFG.lat_min, CFG.lat_max)) &
            (df.Long.between(CFG.lon_min, CFG.lon_max))]
    df = df[df.ServiceID.isin(CFG.active_services)]
    df["day_index"] = day_index
    df["global_start_sec"] = day_index * CFG.seconds_per_day + df["startSec"]
    return df


def _expand_service_chains(events: pd.DataFrame) -> pd.DataFrame:
    """
    افزایش یکنواخت ×۳ ترافیک با «زنجیره‌ی درخواست همتراز» (common/config.py:
    SERVICE_EXPANSION_MAP، افراز ۵×۳ بر اساس تراز deadline). هر رویداد ورودی
    با ServiceID=k، علاوه بر خودش، برای دو عضو دیگر همان سه‌تایی هم یک
    رویداد مشتق (همان BTS/موقعیت، با jitter زمانی کوچک تا کاملاً هم‌زمان
    نباشند) تولید می‌کند - چون هر سه‌تایی دقیقاً ۳ عضو دارد، ضریب تقویت هر
    سرویس بدون استثنا دقیقاً ۳× است (نه نگاشت نامتوازن قبلی).

    عمداً *قطعی* (بدون احتمال fewer از ۱) است تا ضریب ۳× دقیق و یکنواخت
    تضمین شود؛ فقط jitter زمانی برای شکستن هم‌زمانی مصنوعی اضافه می‌شود.
    یک RNG با seed ثابت (CFG.seed) استفاده می‌شود تا بین اجرای الگوریتم‌های
    مختلف روی همان داده (evaluation/compare_runs.py) دقیقاً همان دیتای
    منبسط‌شده بازتولید شود - قابل تکرار و قابل مقایسه.
    """
    if not CFG.enable_service_expansion:
        return events

    rng = np.random.default_rng(CFG.seed)
    extra_frames = []
    for src_sid, group in CFG.service_expansion_map.items():
        siblings = [t for t in group if t != src_sid]
        if not siblings:
            continue
        src_rows = events[events.ServiceID == src_sid]
        if src_rows.empty:
            continue
        for t in siblings:
            derived = src_rows.copy()
            jitter = rng.uniform(0.0, CFG.service_expansion_jitter_sec, size=len(derived))
            derived["ServiceID"] = t
            derived["global_start_sec"] = derived["global_start_sec"] + jitter
            extra_frames.append(derived)

    if not extra_frames:
        return events
    expanded = pd.concat([events] + extra_frames, ignore_index=True)
    return expanded.sort_values("global_start_sec").reset_index(drop=True)


def _apply_traffic_multiplier(events: pd.DataFrame) -> pd.DataFrame:
    """
    *** بازبینی: پشتیبانی از ضریب مستقل هر سرویس (common/config.py:
    TRAFFIC_MULTIPLIER_PER_SERVICE) علاوه‌بر ضریب سراسری قبلی. دلیل: با
    یک ضریب یکسان، سرویس‌های خیلی سریع (svc1..10) یا هرگز به رپلیکای
    چندگانه نمی‌رسند یا برای رسیدنشان باید ضریبی انتخاب شود که سرویس‌های
    سنگین (svc11..15، همان ضریب) را به‌شدت overload می‌کند - چون
    Erlang = rate × exec_time و exec_time این دو گروه سه مرتبه‌ی بزرگی
    فرق دارد.

    اگر CFG.enable_per_service_multiplier=True باشد، برای هر سرویس ابتدا
    ضریب اختصاصی‌اش (اگر در دیکشنری بود) وگرنه CFG.traffic_multiplier
    سراسری (fallback) استفاده می‌شود. اگر این پرچم خاموش باشد، رفتار
    دقیقاً همان نسخه‌ی قبلی (یک ضریب سراسری روی کل دیتافریم) است - یعنی
    برای هرکس این قابلیت را روشن نکرده هیچ تغییری در نتیجه ایجاد نمی‌شود.
    """
    per_service = CFG.traffic_multiplier_per_service if CFG.enable_per_service_multiplier else {}
    default_factor = CFG.traffic_multiplier

    if not per_service and default_factor <= 1.0:
        return events

    rng = np.random.default_rng(CFG.seed + 1)

    def _expand(group: pd.DataFrame, factor: float) -> pd.DataFrame:
        if factor <= 1.0:
            return group
        n_full = int(factor)
        frac = factor - n_full
        frames = [group]
        for _ in range(n_full - 1):
            dup = group.copy()
            jitter = rng.uniform(0.0, CFG.traffic_multiplier_jitter_sec, size=len(dup))
            dup["global_start_sec"] = dup["global_start_sec"] + jitter
            frames.append(dup)
        if frac > 1e-9:
            sample_seed = int(rng.integers(0, 2**31 - 1))
            sample = group.sample(frac=frac, random_state=sample_seed).copy()
            jitter = rng.uniform(0.0, CFG.traffic_multiplier_jitter_sec, size=len(sample))
            sample["global_start_sec"] = sample["global_start_sec"] + jitter
            frames.append(sample)
        return pd.concat(frames, ignore_index=True)

    if not per_service:
        combined = _expand(events, default_factor)
    else:
        out_frames = []
        for sid, group in events.groupby("ServiceID"):
            factor = per_service.get(sid, default_factor)
            out_frames.append(_expand(group, factor))
        combined = pd.concat(out_frames, ignore_index=True)

    return combined.sort_values("global_start_sec").reset_index(drop=True)


def load_timeline(filenames: list[str]) -> pd.DataFrame: 
    frames = [_load_one_day(f, i) for i, f in enumerate(filenames)]
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values("global_start_sec").reset_index(drop=True)
    events = _expand_service_chains(events)
    events = _apply_traffic_multiplier(events)
    return events


def load_train() -> pd.DataFrame:
    """شنبه‌های هفته‌ی ۱ تا ۳ -> یک تایم‌لاین پیوسته‌ی سه‌روزه (day_index=0,1,2)."""
    return load_timeline(list(CFG.train_files))


def load_test() -> pd.DataFrame:
    """شنبه‌ی هفته‌ی ۴ -> تایم‌لاین یک‌روزه (day_index=0 دوباره از صفر، مستقل از train)."""
    return load_timeline([CFG.test_file])


if __name__ == "__main__":
    train = load_train()
    test = load_test()
    print(f"Train: {len(train)} رویداد، بازه‌ی global_start_sec: "
          f"{train.global_start_sec.min()} تا {train.global_start_sec.max()} "
          f"({train.global_start_sec.max()/86400:.2f} روز)")
    print(f"Test:  {len(test)} رویداد، بازه‌ی global_start_sec: "
          f"{test.global_start_sec.min()} تا {test.global_start_sec.max()}")
    print("توزیع day_index در train:")
    print(train.day_index.value_counts().sort_index())