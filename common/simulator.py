"""
حلقه‌ی اصلی شبیه‌سازی چند-سرویسی event-level.

*** توالی بازطراحی‌شده (چون حالا server scaler مبتنی بر utilization چرخه‌ی
قبل تصمیم می‌گیرد، نه on-demand):
    ۱) ready = server_scaler.ready_mask()  -- سرورهای روشنِ آماده‌ی همین چرخه
    ۲) هر service scaler چیدمانش را *فقط* محدود به `ready` پیشنهاد می‌دهد
       (allowed_servers=ready؛ نمی‌تواند سرور خاموش را کاندید کند).
    ۳) dispatcher رویدادهای واقعی همین چرخه را طبق این چیدمان مسیریابی می‌کند.
    ۴) بار کل هر سرور (مجموع همه‌ی سرویس‌ها) از خروجی dispatcher محاسبه و به
       server_scaler داده می‌شود تا utilization را بسنجد و برای چرخه‌ی *بعد*
       سرور اضافه/کم کند (کنترل واکنشی با یک چرخه تاخیر - مثل HPA واقعی).
    ۵) انرژی مصرفی این چرخه محاسبه و به‌صورت تجمعی جمع می‌شود.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from common.config import CFG
from common.data_loader import cycle_loads_for_service
from server_scaler.server_scaler import ServerScaler
from dispatcher.dispatcher import Dispatcher
from metrics.metrics_engine import cycle_summary, system_metrics


def _cycle_energy_joule(outcomes: pd.DataFrame, ready_mask: np.ndarray,
                         effective_placements: dict, service_profile: dict) -> float:
    """
    *** مدل انرژی سه‌جزئی:
      ۱) idle خودِ سرور فیزیکی روشن، در کل مدت چرخه.
      ۲) *** overhead نگه‌داشتن هر replica زنده (متناسب با تعداد کل replica،
         نه فقط تعداد سرور روشن - بدون این، الگوریتمی که replica بیشتری نگه
         می‌دارد بدون هیچ هزینه‌ای سود QoS می‌برد).
      ۳) توان پردازش هر درخواستِ واقعاً سرویس‌داده‌شده، *مقیاس‌شده با
         ServiceResource* همان سرویس (نه یک عدد ثابت برای همه).
    """
    idle_energy = ready_mask.sum() * CFG.SERVER_IDLE_POWER_W * CFG.TAU

    total_replicas = sum(int(p.sum()) for p in effective_placements.values())
    replica_overhead_energy = total_replicas * CFG.REPLICA_IDLE_POWER_W * CFG.TAU

    served = outcomes[outcomes.served_by != -1]
    if len(served):
        resource_units = served.ServiceID.map(lambda sid: service_profile.get(sid, {}).get("resource", 1.0))
        exec_times = served.ServiceID.map(lambda sid: service_profile.get(sid, {}).get("execution_time_sec", 0.0))
        processing_energy = (CFG.SERVER_ACTIVE_POWER_PER_RESOURCE_W * resource_units * exec_times).sum()
    else:
        processing_energy = 0.0

    return float(idle_energy + replica_overhead_energy + processing_energy)


def run_multi_service_simulation(scaler_factory, n_gateways: int, n_servers: int,
                                  events: pd.DataFrame, L: np.ndarray, D_km: np.ndarray,
                                  service_profile: dict, co_per_cycle: float = CFG.CO_PER_CYCLE):
    """
    scaler_factory : callable که با یک آرگومان service_id یک نمونه‌ی تازه از
                      یک service scaler می‌سازد (هر سرویس نمونه‌ی مستقل خودش
                      را دارد). Greedy/HPA/Voila آرگومان را نادیده می‌گیرند؛
                      PPOScaler از آن برای بارگذاری پروفایل همان سرویس استفاده می‌کند.
    خروجی: (service_cycle_df, system_cycle_df, server_scaler)
    """
    server_scaler = ServerScaler(n_servers)
    dispatcher = Dispatcher(L, D_km, CFG.LO_MS, service_profile)
    services = list(CFG.ACTIVE_SERVICES)
    service_scalers = {sid: scaler_factory(sid) for sid in services}
    loads_by_service = {sid: cycle_loads_for_service(events, sid, n_gateways) for sid in services}
    co_per_service = {sid: co_per_cycle for sid in services}

    service_rows, system_rows = [], []
    cumulative_energy = 0.0

    for cycle in range(CFG.N_CYCLES):
        ready_mask = server_scaler.ready_mask()

        # *** هر service scaler فقط مجاز به استفاده از سرورهای روشنِ همین چرخه است
        effective = {sid: service_scalers[sid].decide(loads_by_service[sid][cycle], ready_mask)
                     for sid in services}

        cyc_events = events[events.cycle == cycle]
        outcomes, replica_load = dispatcher.process_cycle(cyc_events, effective, ready_mask, co_per_service)

        service_rows.extend(cycle_summary(cycle, outcomes, effective, ready_mask))

        cumulative_energy += _cycle_energy_joule(outcomes, ready_mask, effective, service_profile)

        sm = system_metrics(outcomes, ready_mask, replica_load, effective)
        sm["cycle"] = cycle
        sm["total_replicas"] = int(sum(p.sum() for p in effective.values()))
        sm["cumulative_energy_joule"] = cumulative_energy
        sm["server_utilization"] = server_scaler.last_utilization
        system_rows.append(sm)

        # *** بار کل هر سرور (همه‌ی سرویس‌ها) را برای تصمیم چرخه‌ی بعد به server scaler بده
        load_per_server = np.zeros(n_servers)
        for arr in replica_load.values():
            load_per_server += arr
        server_scaler.update(load_per_server)

    return pd.DataFrame(service_rows), pd.DataFrame(system_rows), server_scaler
