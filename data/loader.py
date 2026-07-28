"""
data/loader.py
خواندن CSVها، فیلتر منطقه‌ی جغرافیایی، و اعمال آفست روزانه برای ساخت
تایم‌لاین پیوسته (بخش ۱.۳ سند: global_start_sec = day_index * 86400 + startSec).
"""

from __future__ import annotations
import os
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


def load_timeline(filenames: list[str]) -> pd.DataFrame:
    """
    خروجی: یک DataFrame واحد، مرتب‌شده بر اساس global_start_sec، شامل تمام
    فایل‌های ورودی با آفست روزانه‌ی صحیح (بدون تداخل بین روزها).
    """
    frames = [_load_one_day(f, i) for i, f in enumerate(filenames)]
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values("global_start_sec").reset_index(drop=True)
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
