"""
simulator/engine.py
موتور discrete-event. طبق بخش ۱۰ سند باید از simpy استفاده می‌شد؛ چون simpy در
محیط توسعه/اجرای من قابل‌نصب نبود (بدون دسترسی شبکه به pypi برای این پکیج
خاص)، یک موتور سبک و کاملاً معادل با heapq پیاده‌سازی شده که همان مدل
رویداد-محور را با اولویت زمانی انجام می‌دهد. اگر روی سیستم خودتان simpy در
دسترس بود و ترجیح می‌دهید از آن استفاده شود، فقط همین فایل باید بازنویسی
شود؛ common/، algorithms/، data/ نیازی به تغییر ندارند.
"""

from __future__ import annotations
import heapq
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from common.config import CFG
from common.geo import haversine_km, network_delay_ms
from common.models import Server, Replica, Request, ServerState, ReplicaState, RequestStatus
from common.metrics import MetricsCollector
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionActionType
from simulator.events import Event, EventType


class SimulationEngine:
    def __init__(self, events_df: pd.DataFrame, algorithm: AlgorithmBase,
                 algorithm_name: str, event_logger=None, verbose: bool = False):
        self.events_df = events_df
        self.algorithm = algorithm
        self.metrics = MetricsCollector(algorithm=algorithm_name)
        self.logger = event_logger
        self.verbose = verbose

        self.servers: Dict[int, Server] = self._init_servers()
        self.replicas_by_service: Dict[int, List[Replica]] = defaultdict(list)

        self._heap: List[Event] = []
        self._seq = 0
        self.now = 0.0
        self._energy_last_update: Dict[int, float] = {sid: 0.0 for sid in self.servers}
        self._low_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}

        # آمار «از آخرین تیک تصمیم» برای متریک‌های ورودی scale_decision
        self._tick_total = defaultdict(int)
        self._tick_rejected = defaultdict(int)
        self._tick_lat_sum = defaultdict(float)
        self._tick_lon_sum = defaultdict(float)
        self._tick_violated = defaultdict(int)
        self._tick_response_times: List[float] = []
        self._energy_at_last_tick = 0.0
        self._last_tick_decisions = {"provision": None, "scale": {}}

        self._request_seq = 0
        self._service_demand_centroid: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    def _init_servers(self) -> Dict[int, Server]:
        servers = {}
        for sid, info in CFG.server_info.items():
            prof = CFG.server_profiles[info["profile"]]
            servers[sid] = Server(id=sid, profile=info["profile"], lat=info["lat"], long=info["long"],
                                   capacity=info["capacity"], p_idle=prof["p_idle"], p_max=prof["p_max"])
        return servers

    def _push(self, time: float, etype: EventType, payload=None):
        self._seq += 1
        heapq.heappush(self._heap, Event(time, self._seq, etype, payload))

    # ------------------------------------------------------------------
    # انرژی: انتگرال‌گیری پله‌ای (piecewise-constant) بین رویدادها
    # ------------------------------------------------------------------
    def _advance_energy_to(self, t: float):
        for sid, s in self.servers.items():
            last = self._energy_last_update[sid]
            if t > last:
                power = s.instantaneous_power_w(last)
                s.cumulative_energy_joule += power * (t - last)
                self._energy_last_update[sid] = t

    # ------------------------------------------------------------------
    # بخش ۴: جایگذاری اولیه
    # ------------------------------------------------------------------
    def _initial_placement(self):
        first_window = self.events_df[self.events_df.global_start_sec <=
                                       self.events_df.global_start_sec.min() + CFG.monitor_window_sec]
        active_bts = list(first_window[["Lat", "Long"]].drop_duplicates().itertuples(index=False, name=None))
        selected = self.algorithm.initial_placement(self.servers, active_bts)

        for sid in selected:
            self._start_server_boot(sid)

        for service_id in CFG.active_services:
            target = self._nearest_capable_server(service_id, selected)
            if target is not None:
                self._place_replica(target, service_id)

    def _nearest_capable_server(self, service_id: int, candidate_ids: List[int]) -> Optional[int]:
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [self.servers[sid] for sid in candidate_ids
                      if self.servers[sid].can_host(service_id, cpu)]
        if not candidates:
            return None
        centroid_lat = sum(s.lat for s in candidates) / len(candidates)
        centroid_lon = sum(s.long for s in candidates) / len(candidates)
        candidates.sort(key=lambda s: haversine_km(centroid_lat, centroid_lon, s.lat, s.long))
        return candidates[0].id

    # ------------------------------------------------------------------
    # گذارهای سرور
    # ------------------------------------------------------------------
    def _start_server_boot(self, server_id: int):
        s = self.servers[server_id]
        if s.state != ServerState.OFF:
            return
        s.state = ServerState.BOOTING
        s.boot_started_at = self.now
        s.last_transition_time = self.now
        s.num_boots += 1
        s.cumulative_energy_joule += CFG.e_boot_server_j  # انرژی گذار ثابت
        self.metrics.record_transition("server_boot")
        self._log("server_boot_started", server_id=server_id)
        self._push(self.now + CFG.boot_delay_sec, EventType.SERVER_BOOT_DONE, server_id)

    def _handle_boot_done(self, server_id: int):
        s = self.servers[server_id]
        s.state = ServerState.ACTIVE
        s.last_transition_time = self.now
        self._log("server_active", server_id=server_id)
        # هر رپلیکای STARTING که منتظر روشن‌شدن سرور بود، حالا pod-create واقعی می‌شود
        for r in s.hosted_replicas.values():
            if r.state == ReplicaState.STARTING and r.ready_since is None and r.created_at <= self.now:
                self._schedule_replica_ready(r)

    def _start_server_drain(self, server_id: int) -> bool:
        s = self.servers[server_id]
        if s.state != ServerState.ACTIVE:
            return False

        # بخش ۶.۲: مهاجرت رپلیکاهای تک‌رپلیکایی *قبل* از drain کردن خودشان
        steps = self.algorithm.migration_decision(s, self.servers)
        migrated_services = {step.service_id for step in steps}
        sole_hosted = {
            svc_id for svc_id, r in s.hosted_replicas.items()
            if r.state != ReplicaState.TERMINATED and not any(
                other.id != server_id and svc_id in other.hosted_replicas and
                other.hosted_replicas[svc_id].state != ReplicaState.TERMINATED
                for other in self.servers.values())
        }
        # *** محافظ: اگر migration نتواند مقصدی برای همه‌ی سرویس‌های تک‌رپلیکا
        # پیدا کند، drain را این چرخه لغو کن (به‌جای قطع کامل آن سرویس‌ها).
        # طبق بخش ۶.۲ در این حالت باید یک سرور جدید Boot اضطراری شود؛ فعلاً
        # به‌عنوان راه‌حل ایمن‌تر، drain را عقب می‌اندازیم تا چرخه‌ی بعد با
        # وضعیت ظرفیت جدید دوباره امتحان شود.
        if not sole_hosted.issubset(migrated_services):
            self._log("server_drain_aborted", server_id=server_id,
                      reason="migration_incomplete", unmigrated=list(sole_hosted - migrated_services))
            return False

        s.state = ServerState.DRAINING
        s.drain_started_at = self.now
        s.last_transition_time = self.now
        self._log("server_drain_started", server_id=server_id)

        for step in steps:
            self._place_replica(step.target_server_id, step.service_id)

        for r in list(s.hosted_replicas.values()):
            self._start_replica_drain(r)

        self._push(self.now + CFG.server_drain_grace_sec, EventType.SERVER_DRAIN_DONE, server_id)
        return True

    def _handle_drain_done(self, server_id: int):
        s = self.servers[server_id]
        # اگر هنوز رپلیکای درحال‌تخلیه دارد، صبر بیشتر (نادر، ولی برای امنیت)
        if any(r.state == ReplicaState.DRAINING for r in s.hosted_replicas.values()):
            self._push(self.now + CFG.server_drain_grace_sec, EventType.SERVER_DRAIN_DONE, server_id)
            return
        s.hosted_replicas = {sid: r for sid, r in s.hosted_replicas.items()
                              if r.state != ReplicaState.TERMINATED}
        s.state = ServerState.OFF
        s.last_transition_time = self.now
        s.num_shutdowns += 1
        self.metrics.record_transition("server_shutdown")
        self._log("server_off", server_id=server_id)

    # ------------------------------------------------------------------
    # گذارهای رپلیکا
    # ------------------------------------------------------------------
    def _place_replica(self, server_id: int, service_id: int) -> Optional[Replica]:
        s = self.servers[server_id]
        cpu = CFG.services_info[service_id]["cpu_demand"]
        if not s.can_host(service_id, cpu):
            return None
        svc = CFG.services_info[service_id]
        r = Replica(service_id=service_id, server_id=server_id,
                    queue_len=svc["queue_len"], exec_time=svc["exec_time"], created_at=self.now)
        s.hosted_replicas[service_id] = r
        self.replicas_by_service[service_id].append(r)
        self.metrics.record_transition("pod_create")
        s.cumulative_energy_joule += CFG.e_pod_create_j
        self._log("pod_create_started", server_id=server_id, service_id=service_id)
        if s.state == ServerState.ACTIVE:
            self._schedule_replica_ready(r)
        return r

    def _schedule_replica_ready(self, r: Replica):
        self._push(self.now + CFG.pod_startup_delay_sec, EventType.REPLICA_READY,
                    (r.server_id, r.service_id))

    def _handle_replica_ready(self, key):
        server_id, service_id = key
        r = self.servers[server_id].hosted_replicas.get(service_id)
        if r is None or r.state != ReplicaState.STARTING:
            return
        r.state = ReplicaState.READY
        r.ready_since = self.now
        self._log("pod_ready", server_id=server_id, service_id=service_id)

    def _start_replica_drain(self, r: Replica):
        if r.state == ReplicaState.TERMINATED:
            return
        r.state = ReplicaState.DRAINING
        r.drain_started_at = self.now
        self._log("pod_drain_started", server_id=r.server_id, service_id=r.service_id)
        self._push(self.now + CFG.graceful_termination_delay_sec, EventType.REPLICA_TERMINATED,
                    (r.server_id, r.service_id))

    def _handle_replica_terminated(self, key):
        server_id, service_id = key
        s = self.servers[server_id]
        r = s.hosted_replicas.get(service_id)
        if r is None:
            return
        r.state = ReplicaState.TERMINATED
        self.metrics.record_transition("pod_delete")
        self._log("pod_terminated", server_id=server_id, service_id=service_id)
        del s.hosted_replicas[service_id]
        self.replicas_by_service[service_id] = [
            x for x in self.replicas_by_service[service_id] if x is not r]

    # ------------------------------------------------------------------
    # بخش ۳: چرخه‌ی درخواست
    # ------------------------------------------------------------------
    def _handle_arrival(self, row):
        self._request_seq += 1
        req = Request(id=self._request_seq, bts_lat=row.Lat, bts_long=row.Long,
                       service_id=int(row.ServiceID), arrival_time=self.now)
        self._tick_total[req.service_id] += 1
        self._tick_lat_sum[req.service_id] += req.bts_lat
        self._tick_lon_sum[req.service_id] += req.bts_long
        self._log("request_arrived", request_id=req.id, service_id=req.service_id)

        candidates = [r for r in self.replicas_by_service.get(req.service_id, [])
                      if r.is_selectable()]
        chosen = self.algorithm.select_replica(req, candidates, self.servers, self.now)

        if chosen is None:
            if not candidates:
                req.status = RequestStatus.REJECTED_NO_REPLICA
            else:
                req.status = RequestStatus.REJECTED_QUEUE_FULL
            self._tick_rejected[req.service_id] += 1
            self._tick_violated[req.service_id] += 1
            self._finalize_request(req)
            return

        server = self.servers[chosen.server_id]
        distance_km = haversine_km(req.bts_lat, req.bts_long, server.lat, server.long)
        delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
        req._distance_km = distance_km
        req.network_delay_ms = delay_ms
        req.assigned_server_id = server.id

        cold_start_extra = 0.0
        if chosen.ready_since is not None and (self.now - chosen.ready_since) <= CFG.cold_start_window_sec:
            cold_start_extra = CFG.cold_start_penalty_sec

        admit = chosen.try_admit(self.now, cold_start_extra=cold_start_extra)
        if admit is None:
            req.status = RequestStatus.REJECTED_QUEUE_FULL
            self._tick_rejected[req.service_id] += 1
            self._tick_violated[req.service_id] += 1
            self._log("request_rejected", request_id=req.id, service_id=req.service_id,
                      reason="queue_full")
            self._finalize_request(req)
            return

        req.queue_enter_time = admit["queue_enter_time"]
        req.service_start_time = admit["service_start_time"]
        req.service_end_time = admit["service_end_time"]
        req.wait_time_sec = admit["wait_time_sec"]
        # بخش ۳: response_time = ۲×network_delay (رفت+برگشت) + wait + exec (+cold start در exec لحاظ شد)
        req.response_time_sec = (2 * delay_ms / 1000.0) + req.wait_time_sec + \
                                 (req.service_end_time - req.service_start_time)
        req.deadline_violated = req.response_time_sec > CFG.services_info[req.service_id]["deadline"]
        req.status = RequestStatus.COMPLETED
        if req.deadline_violated:
            self._tick_violated[req.service_id] += 1
        self._tick_response_times.append(req.response_time_sec)
        self._log("request_completed", request_id=req.id, service_id=req.service_id,
                  server_id=server.id, response_time_sec=req.response_time_sec)
        self._finalize_request(req)

    def _finalize_request(self, req: Request):
        self.metrics.record_request(req)

    def _log(self, event_type: str, **fields):
        if self.logger is not None:
            self.logger.log(event_type, sim_time=self.now, **fields)

    # ------------------------------------------------------------------
    # بخش ۶ و ۷: تیک تصمیم (هر DECISION_INTERVAL_SEC)
    # ------------------------------------------------------------------
    def peek_snapshot(self) -> dict:
        """
        snapshot لحظه‌ای بدون اجرای منطق تیک تصمیم (فقط خواندن وضعیت فعلی).
        برای observation اولیه‌ی reset() محیط PPO استفاده می‌شود - چون بلافاصله
        بعد از prime() هنوز هیچ اکشنی از عامل نرسیده و نباید تیک واقعی رخ دهد.
        """
        return self._build_metrics_snapshot_readonly()

    def _build_metrics_snapshot_readonly(self) -> dict:
        """مثل _build_metrics_snapshot ولی شمارنده‌های تیک را پاک/تغییر نمی‌دهد."""
        snapshot = {"servers": {}, "services": {}, "global": {}}
        for sid, s in self.servers.items():
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": s.instantaneous_utilization(self.now),
                "free_capacity": s.free_capacity(),
            }
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(r.queue_occupancy(self.now) for r in ready) / len(ready)) if ready else 0.0
            snapshot["services"][svc_id] = {
                "n_replicas": len([r for r in reps if r.state in
                                    (ReplicaState.READY, ReplicaState.STARTING)]),
                "avg_queue_occupancy": avg_occ,
                "queue_len": CFG.services_info[svc_id]["queue_len"],
                "rejection_rate": self._tick_rejected[svc_id] / total,
                "deadline_violation_rate": self._tick_violated[svc_id] / total,
                "recent_arrivals": self._tick_total[svc_id],
                "demand_centroid": self._service_demand_centroid.get(svc_id),
            }
        snapshot["global"] = {"avg_response_time_recent": 0.0, "energy_recent_joule": 0.0,
                               "num_rejected_recent": 0}
        return snapshot

    def _build_metrics_snapshot(self) -> dict:
        snapshot = {"servers": {}, "services": {}, "global": {}}
        for sid, s in self.servers.items():
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": s.instantaneous_utilization(self.now),
                "free_capacity": s.free_capacity(),
            }
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(r.queue_occupancy(self.now) for r in ready) / len(ready)) if ready else 0.0
            if self._tick_total[svc_id] > 0:
                new_lat = self._tick_lat_sum[svc_id] / self._tick_total[svc_id]
                new_lon = self._tick_lon_sum[svc_id] / self._tick_total[svc_id]
                if svc_id in self._service_demand_centroid:
                    old_lat, old_lon = self._service_demand_centroid[svc_id]
                    alpha = 0.3  # میانگین متحرک نمایی برای پایداری بین چرخه‌های کم‌درخواست
                    new_lat = alpha * new_lat + (1 - alpha) * old_lat
                    new_lon = alpha * new_lon + (1 - alpha) * old_lon
                self._service_demand_centroid[svc_id] = (new_lat, new_lon)
            snapshot["services"][svc_id] = {
                "n_replicas": len([r for r in reps if r.state in
                                    (ReplicaState.READY, ReplicaState.STARTING)]),
                "avg_queue_occupancy": avg_occ,
                "queue_len": CFG.services_info[svc_id]["queue_len"],
                "rejection_rate": self._tick_rejected[svc_id] / total,
                "deadline_violation_rate": self._tick_violated[svc_id] / total,
                "recent_arrivals": self._tick_total[svc_id],
                "demand_centroid": self._service_demand_centroid.get(svc_id),
            }
        current_energy = sum(s.cumulative_energy_joule for s in self.servers.values())
        snapshot["global"] = {
            "avg_response_time_recent": (sum(self._tick_response_times) / len(self._tick_response_times))
                                         if self._tick_response_times else 0.0,
            "energy_recent_joule": current_energy - self._energy_at_last_tick,
            "num_rejected_recent": sum(self._tick_rejected.values()),
        }
        self._energy_at_last_tick = current_energy
        return snapshot

    def _apply_provisioning(self, action, snapshot: dict):
        self._last_tick_decisions["provision"] = action
        if action.action == ProvisionActionType.TURN_ON and action.server_id is not None:
            s = self.servers[action.server_id]
            if s.state == ServerState.OFF and not s.in_cooldown(self.now, CFG.cooldown_sec):
                self._start_server_boot(action.server_id)
                self.metrics.record_scale_action("TURN_ON")
        elif action.action == ProvisionActionType.TURN_OFF and action.server_id is not None:
            s = self.servers[action.server_id]
            n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
            sustained = (self._low_util_since[action.server_id] is not None and
                         (self.now - self._low_util_since[action.server_id]) >= CFG.sustain_low_sec)
            # *** محافظ سیستمی: هرگز آخرین سرور فعال را drain نکن.
            if s.state == ServerState.ACTIVE and sustained and n_active > 1 and \
                    not s.in_cooldown(self.now, CFG.cooldown_sec):
                if self._start_server_drain(action.server_id):
                    self.metrics.record_scale_action("TURN_OFF")

    def _apply_scale_decision(self, svc_id: int, decision: ScaleAction):
        self._last_tick_decisions["scale"][svc_id] = decision
        if decision == ScaleAction.SCALE_UP:
            target = self.algorithm.select_placement_server(svc_id, self.servers)
            if target is not None:
                self._place_replica(target, svc_id)
                self.metrics.record_scale_action("SCALE_UP")
        elif decision == ScaleAction.SCALE_DOWN:
            ready = [r for r in self.replicas_by_service.get(svc_id, [])
                     if r.state == ReplicaState.READY]
            if len(ready) > 1:  # حداقل ۱ رپلیکا همیشه باید بماند
                victim = min(ready, key=lambda r: r.queue_occupancy(self.now))
                self._start_replica_drain(victim)
                self.metrics.record_scale_action("SCALE_DOWN")

    def _update_sustain_tracking(self, snapshot: dict):
        for sid, s in self.servers.items():
            if s.state != ServerState.ACTIVE:
                continue
            util = snapshot["servers"][sid]["utilization"]
            if util < CFG.util_scale_down_threshold:
                if self._low_util_since[sid] is None:
                    self._low_util_since[sid] = self.now
            else:
                self._low_util_since[sid] = None

    def _handle_decision_tick(self, external_actions: dict | None = None) -> dict:
        """
        بخش ۶ و ۷: تیک تصمیم. اگر external_actions داده شود (فقط برای آموزش
        PPO از طریق algorithms/ppo/env.py استفاده می‌شود)، به‌جای فراخوانی
        متدهای self.algorithm از آن استفاده می‌شود؛ در غیر این‌صورت رفتار
        عادی (Greedy/Voila/HPA/PPO-inference از طریق ppo_algorithm.py) اجرا
        می‌شود. خروجی: همان metrics_snapshot این تیک (برای ساخت reward/state).
        """
        self.metrics.record_snapshot(self.now, self.servers)
        snapshot = self._build_metrics_snapshot()
        self._update_sustain_tracking(snapshot)
        self._last_tick_decisions = {"provision": None, "scale": {}}

        if external_actions is not None:
            self._apply_provisioning(external_actions["provision"], snapshot)
            for svc_id, decision in external_actions["scale"].items():
                self._apply_scale_decision(svc_id, decision)
        else:
            action = self.algorithm.provision_decision(self.servers, snapshot, self.now)
            self._apply_provisioning(action, snapshot)
            for svc_id in CFG.active_services:
                decision = self.algorithm.scale_decision(svc_id, snapshot)
                self._apply_scale_decision(svc_id, decision)

        self._tick_total.clear()
        self._tick_rejected.clear()
        self._tick_violated.clear()
        self._tick_response_times.clear()
        self._tick_lat_sum.clear()
        self._tick_lon_sum.clear()
        return snapshot

    # ------------------------------------------------------------------
    # اجرای batch عادی (Greedy/Voila/HPA/PPO-inference از طریق ppo_algorithm.py)
    # ------------------------------------------------------------------
    def run(self) -> dict:
        self.prime()
        while True:
            _, done = self.step()
            if done:
                break
        return self.metrics.finalize(self.servers)

    # ------------------------------------------------------------------
    # اجرای قابل‌step (برای algorithms/ppo/env.py هنگام آموزش)
    # ------------------------------------------------------------------
    def prime(self):
        """بارگذاری رویدادهای ورود + جایگذاری اولیه؛ باید یک‌بار قبل از step() فراخوانی شود."""
        for row in self.events_df.itertuples(index=False):
            self._push(float(row.global_start_sec), EventType.REQUEST_ARRIVAL, row)
        start_time = float(self.events_df.global_start_sec.min()) if len(self.events_df) else 0.0
        max_time = float(self.events_df.global_start_sec.max()) if len(self.events_df) else 0.0
        self.now = start_time
        self._initial_placement()
        self._push(start_time, EventType.DECISION_TICK)
        self._cutoff = max_time + CFG.decision_interval_sec + CFG.server_drain_grace_sec

    def step(self, external_actions: dict | None = None):
        """
        رویدادها را تا رسیدن به تیک تصمیم *بعدی* پردازش می‌کند.
        خروجی: (metrics_snapshot این تیک یا None اگر پایان یافت, done: bool)
        """
        while self._heap:
            ev = heapq.heappop(self._heap)
            if ev.time > self._cutoff:
                return None, True
            self._advance_energy_to(ev.time)
            self.now = ev.time

            if ev.type == EventType.REQUEST_ARRIVAL:
                self._handle_arrival(ev.payload)
            elif ev.type == EventType.DECISION_TICK:
                snapshot = self._handle_decision_tick(external_actions)
                self._push(self.now + CFG.decision_interval_sec, EventType.DECISION_TICK)
                return snapshot, False
            elif ev.type == EventType.SERVER_BOOT_DONE:
                self._handle_boot_done(ev.payload)
            elif ev.type == EventType.SERVER_DRAIN_DONE:
                self._handle_drain_done(ev.payload)
            elif ev.type == EventType.REPLICA_READY:
                self._handle_replica_ready(ev.payload)
            elif ev.type == EventType.REPLICA_TERMINATED:
                self._handle_replica_terminated(ev.payload)
        return None, True
