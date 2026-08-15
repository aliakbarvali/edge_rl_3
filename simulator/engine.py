"""
simulator/engine.py


"""

from __future__ import annotations
import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from common.config import CFG, compute_exec_time_sec
from common.geo import haversine_km, network_delay_ms
from common.models import Server, Replica, Request, ServerState, ReplicaState, RequestStatus
from common.metrics import MetricsCollector
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionActionType
from simulator.events import Event, EventType
from collections import deque
from common.rolling_signal import RollingRejectionTracker
 
class SimulationEngine:
    def __init__(self, events_df: pd.DataFrame, algorithm: AlgorithmBase,
                 algorithm_name: str, event_logger=None, verbose: bool = False):
        self.events_df = events_df
        self.algorithm = algorithm
        self.metrics = MetricsCollector(algorithm=algorithm_name)
        self.logger = event_logger
        self.verbose = verbose

        self._rolling_rejection = RollingRejectionTracker()
        self.servers: Dict[int, Server] = self._init_servers()
        self.replicas_by_service: Dict[int, List[Replica]] = defaultdict(list)

        self._heap: List[Event] = []
        self._seq = 0
        self.now = 0.0
        self._energy_last_update: Dict[int, float] = {sid: 0.0 for sid in self.servers}
        self._low_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}

        self._tick_total = defaultdict(int)
        self._tick_rejected = defaultdict(int)
        self._tick_lat_sum = defaultdict(float)
        self._tick_lon_sum = defaultdict(float)
        self._tick_violated = defaultdict(int)
        self._tick_response_times: List[float] = []
        self._tick_proximity_violated = defaultdict(int)
        
        self._energy_at_last_tick = 0.0
        self._last_tick_decisions = {"provision": None, "scale": {}}
        # *** رفع BUG-B: _last_tick_decisions همان چیزی است که الگوریتم
        # *پیشنهاد* داده (قبل از بررسی گیت‌های cooldown/ظرفیت/...)، نه
        # چیزی که واقعاً *اعمال* شده. اگر BC warm-start از روی همین دیکشنری
        # عمل کند، وقتی مثلاً Greedy یک SCALE_UP پیشنهاد می‌دهد که به‌خاطر
        # cooldown رد می‌شود، مدل یاد می‌گیرد در آن state دقیقاً همان
        # SCALE_UP (که رد شد) را انجام دهد - یعنی از رفتار واقعی معلم، نه
        # رفتار پیشنهادی‌اش، تقلید نمی‌کند. این دیکشنری‌ی جدا فقط اکشن‌هایی
        # را نگه می‌دارد که applied=True شده‌اند (پیش‌فرض NO_CHANGE/None).
        self._last_tick_applied_actions = {"provision": None, "scale": {}}

        self._request_seq = 0
        self._service_demand_centroid: Dict[int, tuple] = {}

        self._pending_migrations: Dict[Tuple[int, int], int] = {}
        self._emergency_boot_for_service: Dict[int, int] = {}
        self._high_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}

     
        self._service_last_scale_time: Dict[int, float] = {sid: -1e18 for sid in CFG.active_services}
        self._service_last_scale_up_time: Dict[int, float] = {sid: -1e18 for sid in CFG.active_services}
        self._recent_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self._util_at_window_start: Dict[int, float] = {sid: 0.0 for sid in self.servers}
        self._util_window_start_time: float = 0.0
    
    def _init_servers(self) -> Dict[int, Server]:
        servers = {}
        for sid, info in CFG.server_info.items():
            prof = CFG.server_profiles[info["profile"]]
            servers[sid] = Server(id=sid, profile=info["profile"], lat=info["lat"], long=info["long"],
                                   capacity=info["capacity_mips"], p_idle=prof["p_idle"], p_max=prof["p_max"])
        return servers

    def _push(self, time: float, etype: EventType, payload=None):
        self._seq += 1
        heapq.heappush(self._heap, Event(time, self._seq, etype, payload))

    def _advance_energy_to(self, t: float):
        for sid, s in self.servers.items():
            last = self._energy_last_update[sid]
            if t > last:
                util_at_last = s.instantaneous_utilization(last)
                power = s.instantaneous_power_w(last)
                s.cumulative_energy_joule += power * (t - last)
                s.cumulative_busy_cpu_seconds += util_at_last * s.capacity * (t - last)
                self._energy_last_update[sid] = t 

    
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
        cpu = CFG.services_info[service_id]["resource_mips"]
        candidates = [self.servers[sid] for sid in candidate_ids
                      if self.servers[sid].can_host(service_id, cpu)]
        if not candidates:
            return None
        centroid_lat = sum(s.lat for s in candidates) / len(candidates)
        centroid_lon = sum(s.long for s in candidates) / len(candidates)
        candidates.sort(key=lambda s: haversine_km(centroid_lat, centroid_lon, s.lat, s.long))
        return candidates[0].id


    def _trigger_emergency_boot(self, unmigrated_services: set, draining_server: Server):

        off_servers = [x for x in self.servers.values() if x.state == ServerState.OFF]
        reserved_cpu: Dict[int, int] = defaultdict(int)
        for svc_id in unmigrated_services:
            if svc_id in self._emergency_boot_for_service:
                continue  
            cpu = CFG.services_info[svc_id]["resource_mips"]
            candidates = [x for x in off_servers if x.capacity - reserved_cpu[x.id] >= cpu]
            if not candidates:
                continue 
            candidates.sort(key=lambda x: haversine_km(
                draining_server.lat, draining_server.long, x.lat, x.long))
            target = candidates[0]
            reserved_cpu[target.id] += cpu
            self._emergency_boot_for_service[svc_id] = target.id 
            self._start_server_boot(target.id, is_emergency=True) 
            self._log("emergency_boot_triggered", server_id=target.id, service_id=svc_id,
                      source_server_id=draining_server.id, reason="migration_target_unavailable")
            

    def _start_server_boot(self, server_id: int, is_emergency: bool = False):
        s = self.servers[server_id]
        if s.state != ServerState.OFF:
            return
        s.state = ServerState.BOOTING
        s.is_emergency_boot = is_emergency  
        s.boot_started_at = self.now
        s.last_transition_time = self.now
        s.num_boots += 1
        s.cumulative_energy_joule += CFG.e_boot_server_j  
        self.metrics.record_transition("server_boot")
        self._log("server_boot_started", server_id=server_id, is_emergency=is_emergency)
        self._push(self.now + CFG.boot_delay_sec, EventType.SERVER_BOOT_DONE, server_id)


    def _handle_boot_done(self, server_id: int):
        s = self.servers[server_id]
        s.state = ServerState.ACTIVE
        s.last_transition_time = self.now
        self._log("server_active", server_id=server_id)
        for r in s.hosted_replicas.values():
            if r.state == ReplicaState.STARTING and r.ready_since is None and r.created_at <= self.now:
                self._schedule_replica_ready(r)

        rescued = [svc_id for svc_id, target_id in self._emergency_boot_for_service.items()
                   if target_id == server_id]
        for svc_id in rescued:
            del self._emergency_boot_for_service[svc_id]
            self._log("emergency_boot_completed", server_id=server_id, service_id=svc_id)
            
            
    def _start_server_drain(self, server_id: int) -> bool:
        s = self.servers[server_id]
        if s.state != ServerState.ACTIVE:
            return False
        steps = self.algorithm.migration_decision(s, self.servers)

        reserved_cpu: Dict[int, int] = defaultdict(int)
        valid_steps = []
        for step in steps:
            target = self.servers[step.target_server_id]
            cpu = CFG.services_info[step.service_id]["resource_mips"]
            if target.free_capacity() - reserved_cpu[target.id] >= cpu:
                reserved_cpu[target.id] += cpu
                valid_steps.append(step)
            else:
                self._log("migration_step_dropped", service_id=step.service_id,
                          from_server_id=server_id, to_server_id=step.target_server_id,
                          reason="target_capacity_overcommitted")
        steps = valid_steps

        migrated_services = {step.service_id for step in steps}
        sole_hosted = {
            svc_id for svc_id, r in s.hosted_replicas.items()
            if r.state != ReplicaState.TERMINATED and not any(
                other.id != server_id and svc_id in other.hosted_replicas and
                other.hosted_replicas[svc_id].state != ReplicaState.TERMINATED
                for other in self.servers.values())
        }
      
        unmigrated = sole_hosted - migrated_services
        if unmigrated:
            self._trigger_emergency_boot(unmigrated, s)
            self._log("server_drain_aborted", server_id=server_id,
                      reason="migration_incomplete", unmigrated=list(unmigrated))
            return False
        s.state = ServerState.DRAINING
        s.drain_started_at = self.now
        s.last_transition_time = self.now
        self._log("server_drain_started", server_id=server_id)

        for step in steps:
            self._log("migration_started", service_id=step.service_id,
                      from_server_id=server_id, to_server_id=step.target_server_id)
            placed = self._place_replica(step.target_server_id, step.service_id)
            if placed is None:
              
                self._log("migration_placement_failed", service_id=step.service_id,
                          from_server_id=server_id, to_server_id=step.target_server_id)
                continue
            self._pending_migrations[(step.target_server_id, step.service_id)] = server_id
    

        for r in list(s.hosted_replicas.values()):
            if r.service_id in migrated_services:
                continue
            self._start_replica_drain(r)

        self._push(self.now + CFG.server_drain_grace_sec, EventType.SERVER_DRAIN_DONE, server_id)
        return True

    def _handle_drain_done(self, server_id: int):
        s = self.servers[server_id]
        if any(r.state != ReplicaState.TERMINATED for r in s.hosted_replicas.values()):
            self._push(self.now + CFG.server_drain_grace_sec, EventType.SERVER_DRAIN_DONE, server_id)
            return
        s.hosted_replicas = {sid: r for sid, r in s.hosted_replicas.items()
                              if r.state != ReplicaState.TERMINATED}
        s.state = ServerState.OFF
        s.last_transition_time = self.now
        s.num_shutdowns += 1
        self.metrics.record_transition("server_shutdown")
        self._log("server_off", server_id=server_id)


    def _place_replica(self, server_id: int, service_id: int) -> Optional[Replica]:
        s = self.servers[server_id]
        cpu = CFG.services_info[service_id]["resource_mips"]
        centroid = self._service_demand_centroid.get(service_id)
        bts_lat, bts_long = centroid if centroid else (None, None)
        if not s.can_host(service_id, cpu, bts_lat=bts_lat, bts_long=bts_long):
            return None
        svc = CFG.services_info[service_id]
        
        r = Replica(service_id=service_id, server_id=server_id,
                    queue_len=svc["queue_len"],
                    exec_time=compute_exec_time_sec(service_id, CFG.server_profiles[s.profile]["mips_per_core"]), created_at=self.now)
                   

        
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


        pending_key = (server_id, service_id)
        source_server_id = self._pending_migrations.pop(pending_key, None)
        if source_server_id is not None:
            source_server = self.servers.get(source_server_id)
            old_replica = source_server.hosted_replicas.get(service_id) if source_server else None
            if old_replica is not None and old_replica.state not in (ReplicaState.TERMINATED,
                                                                       ReplicaState.DRAINING):
                self._start_replica_drain(old_replica)
            self._log("migration_completed", server_id=source_server_id, service_id=service_id,
                      target_server_id=server_id)

    def _start_replica_drain(self, r: Replica):
        if r.state == ReplicaState.TERMINATED:
            return
        r.state = ReplicaState.DRAINING
        r.drain_started_at = self.now
        self._log("pod_drain_started", server_id=r.server_id, service_id=r.service_id)

        drain_wait = max(CFG.graceful_termination_delay_sec, r.available_at - self.now)
        self._push(self.now + drain_wait, EventType.REPLICA_TERMINATED,
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


    def _handle_arrival(self, row):

        self._request_seq += 1
        req = Request(id=self._request_seq, bts_lat=row.Lat, bts_long=row.Long,
                       service_id=int(row.ServiceID), arrival_time=self.now)
        self._tick_total[req.service_id] += 1
        self._tick_lat_sum[req.service_id] += req.bts_lat
        self._tick_lon_sum[req.service_id] += req.bts_long

        self._recent_positions[req.service_id].append((req.bts_lat, req.bts_long))
        self._log("request_arrived", request_id=req.id, service_id=req.service_id)
 
        dispatcher_lat = sum(s.lat for s in self.servers.values()) / len(self.servers)   # ≈ 31.185
        dispatcher_lon = sum(s.long for s in self.servers.values()) / len(self.servers)  # ≈ 121.431
 
        distance_to_dispatcher_km = haversine_km(req.bts_lat, req.bts_long, 
                                                  dispatcher_lat, dispatcher_lon)
        # *** رفت‌وبرگشت واقعی: BTS باید منتظر پاسخ دیسپچر (آدرس سرور مقصد)
        # بماند قبل از این‌که بتواند به سرویس واقعی وصل شود - دقیقاً مثل
        # k8s_adapter/worker_service/bts_simulator.py که واقعاً یک POST به
        # dispatcher_api می‌زند و منتظر جواب HTTP آن می‌ماند (رفت‌وبرگشت
        # واقعی روی شبکه)، نه یک ارسال یک‌طرفه. بدون این ضرب‌در۲، این تأخیر
        # با تأخیر شبکه‌ی BTS<->سرور (که پایین‌تر با 2*delay_ms محاسبه
        # می‌شود) ناهم‌خوان می‌ماند.
        one_way_dispatch_delay_ms = (CFG.base_latency_ms
                                      + CFG.k_ms_per_km * distance_to_dispatcher_km
                                      + CFG.dispatch_overhead_ms)
        routing_delay_ms = 2 * one_way_dispatch_delay_ms
        req.routing_delay_sec = routing_delay_ms / 1000.0
        self._push(self.now + req.routing_delay_sec, EventType.REQUEST_ROUTED, req)

    def _handle_routed(self, req: Request):

        candidates = [r for r in self.replicas_by_service.get(req.service_id, [])
                      if r.is_selectable()]
        chosen = self.algorithm.select_replica(req, candidates, self.servers, self.now)

        if chosen is None:
            if not candidates:
                req.status = RequestStatus.REJECTED_NO_REPLICA
                self._log("request_rejected", request_id=req.id, service_id=req.service_id,
                          reason="no_replica")
            else:
                req.status = RequestStatus.REJECTED_QUEUE_FULL
                self._log("request_rejected", request_id=req.id, service_id=req.service_id,
                          reason="queue_full")
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

        # *** رفع رگرسیون: با محدوده‌ی جغرافیایی فعلی پروژه (LAT/LON_MIN..MAX)
        # بیشینه‌ی فاصله‌ی ممکن BTS<->سرور حدود ۱۸۲ کیلومتر است، یعنی بیشینه‌ی
        # delay_ms یک‌طرفه فقط ~5.6ms می‌شود - همیشه کمتر از PROXIMITY_L0_MS=7.0
        # است. اگر اینجا delay_ms (یک‌طرفه) به‌جای 2*delay_ms (رفت‌وبرگشت) با
        # آستانه مقایسه شود، این شرط عملاً هرگز True نمی‌شود و سیگنال
        # proximity-violation که VOILA برای scale-up جغرافیایی استفاده می‌کند
        # کاملاً خاموش می‌ماند. آستانه باید همان‌طور که در نسخه‌ی اصلی بود، روی
        # تأخیر رفت‌وبرگشت (RTT) سنجیده شود.
        if 2 * delay_ms >= CFG.proximity_l0_ms:
            self._tick_proximity_violated[req.service_id] += 1
        
        
         
        self._log("request_routed", request_id=req.id, service_id=req.service_id,
            server_id=server.id, distance_km=distance_km, network_delay_ms=delay_ms,
            routing_delay_sec=req.routing_delay_sec)
        cold_start_extra = 0.0
        
        from common.config import compute_cold_start_penalty_sec,compute_cold_start_window_sec
        window_sec = compute_cold_start_window_sec(req.service_id, CFG.server_profiles[server.profile]["mips_per_core"])
        
        if chosen.ready_since is not None and (self.now - chosen.ready_since) <= window_sec:
    
            cold_start_extra = compute_cold_start_penalty_sec(
                req.service_id, CFG.server_profiles[server.profile]["mips_per_core"])
        admit = chosen.try_admit(self.now, cold_start_extra=cold_start_extra)
        if admit is None:
   
            self._log("unexpected_admit_race", request_id=req.id, service_id=req.service_id,
                      server_id=chosen.server_id)
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
        self._log("request_queued", request_id=req.id, service_id=req.service_id,
                  server_id=server.id, wait_time_sec=req.wait_time_sec)

        self._push(req.service_end_time, EventType.ENERGY_RESYNC, None)
        
        req.response_time_sec = (req.routing_delay_sec + (2.0 * delay_ms / 1000.0) + 
                                 req.wait_time_sec + (req.service_end_time - req.service_start_time))
        req.deadline_violated = req.response_time_sec > CFG.services_info[req.service_id]["deadline"]
        req.status = RequestStatus.COMPLETED
        if req.deadline_violated:
            self._tick_violated[req.service_id] += 1
 
        self._tick_response_times.append(req.response_time_sec)
        self._log("request_completed", request_id=req.id, service_id=req.service_id,
                server_id=server.id, response_time_sec=req.response_time_sec,
                distance_km=distance_km, network_delay_ms=delay_ms,
                deadline_violated=req.deadline_violated) 
        self._finalize_request(req)
    def _finalize_request(self, req: Request):
        self.metrics.record_request(req)

    def _log(self, event_type: str, **fields):
        if self.logger is not None:
            self.logger.log(event_type, sim_time=self.now, **fields)


    def peek_snapshot(self) -> dict:

        return self._build_metrics_snapshot_readonly()

    def _build_metrics_snapshot_readonly(self) -> dict:
        snapshot = {"servers": {}, "services": {}, "global": {}}
        n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
        for sid, s in self.servers.items():
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": s.instantaneous_utilization(self.now),
                "free_capacity": s.free_capacity(),
                "provision_cooldown_active": s.in_cooldown(self.now, CFG.cooldown_sec),
                "min_active_duration_met": (self.now - s.last_transition_time) >= CFG.min_active_duration_sec,
                "is_last_active_server": (s.state == ServerState.ACTIVE and n_active <= 1),
            }
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            mature_ready = [r for r in ready
                             if (self.now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(r.queue_occupancy(self.now) for r in ready) / len(ready)) if ready else 0.0
            snapshot["services"][svc_id] = {
                "n_replicas": len([r for r in reps if r.state in
                                    (ReplicaState.READY, ReplicaState.STARTING)]),
                "n_ready_replicas": len(ready),
                "n_mature_ready_replicas": len(mature_ready), 
                "avg_queue_occupancy": avg_occ,
                "queue_len": CFG.services_info[svc_id]["queue_len"],
                "rejection_rate": self._tick_rejected[svc_id] / total,
                "deadline_violation_rate": self._tick_violated[svc_id] / total,
                "recent_arrivals": self._tick_total[svc_id],
                "demand_centroid": self._service_demand_centroid.get(svc_id),
                "proximity_violation_rate": self._tick_proximity_violated[svc_id] / total,
                "scale_cooldown_active": (self.now - self._service_last_scale_time[svc_id]) < CFG.cooldown_sec,
            }
        snapshot["global"] = {"avg_response_time_recent": 0.0, "energy_recent_joule": 0.0,
                               "num_rejected_recent": 0}
        return snapshot
        
    def _build_metrics_snapshot(self) -> dict:
        snapshot = {"servers": {}, "services": {}, "global": {}}
        window_elapsed = self.now - self._util_window_start_time
        n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
        for sid, s in self.servers.items():
            if window_elapsed > 1e-9:
                avg_util = ((s.cumulative_busy_cpu_seconds - self._util_at_window_start[sid])
                            / (s.capacity * window_elapsed)) if s.capacity > 0 else 0.0
            else:
                avg_util = s.instantaneous_utilization(self.now)
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": avg_util,
                "free_capacity": s.free_capacity(),
                "provision_cooldown_active": s.in_cooldown(self.now, CFG.cooldown_sec),
                "min_active_duration_met": (self.now - s.last_transition_time) >= CFG.min_active_duration_sec,
                "is_last_active_server": (s.state == ServerState.ACTIVE and n_active <= 1),
            }
        
        def _medoid(points):
            if not points:
                return None
            best, best_cost = points[0], float("inf")
            for p in points:
                cost = sum(haversine_km(p[0], p[1], q[0], q[1]) for q in points)
                if cost < best_cost:
                    best, best_cost = p, cost
            return best
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            mature_ready = [r for r in ready
                             if (self.now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
            total = max(self._tick_total[svc_id], 1)
            self._rolling_rejection.push_tick(
                svc_id, self._tick_rejected[svc_id], self._tick_total[svc_id])
            rolling_rate = self._rolling_rejection.rolling_rejection_rate(svc_id)
            avg_occ = (sum(r.queue_occupancy(self.now) for r in ready) / len(ready)) if ready else 0.0
            if self._tick_total[svc_id] > 0: 
                self._service_demand_centroid[svc_id] = _medoid(list(self._recent_positions[svc_id]))
            snapshot["services"][svc_id] = {
                "n_replicas": len([r for r in reps if r.state in
                                    (ReplicaState.READY, ReplicaState.STARTING)]),
                "n_ready_replicas": len(ready),
                "n_mature_ready_replicas": len(mature_ready), 
                "avg_queue_occupancy": avg_occ,
                "queue_len": CFG.services_info[svc_id]["queue_len"],
                "rejection_rate": self._tick_rejected[svc_id] / total,
                "deadline_violation_rate": self._tick_violated[svc_id] / total,
                "rejection_rate_rolling": rolling_rate, 
                "recent_arrivals": self._tick_total[svc_id],
                "demand_centroid": self._service_demand_centroid.get(svc_id),
                "proximity_violation_rate": self._tick_proximity_violated[svc_id] / total,
                "scale_cooldown_active": (self.now - self._service_last_scale_time[svc_id]) < CFG.cooldown_sec,
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
        applied = False
        skip_reason = None
        via_capacity_starved_only = None
        turn_on_necessary = (self._any_active_server_sustained_overloaded()
                              or self._any_service_capacity_starved(snapshot))
        turn_off_opportunity = self._any_active_server_sustained_underloaded()

        # «هر الگوریتم آزاد است تصمیم provisioning خودش را فوری (فقط با رعایت cooldown/min_active_duration) اعمال کند، بدون گیت مشترک sustain-tracking؛ این گیت صرفاً برای امکان محدودسازی انتخابی یک الگوریتم خاص در آینده نگه داشته شده و پیش‌فرض آن اکنون True است.»
        bypass = getattr(self.algorithm, "bypass_sustain_gate", False)

        if action.action == ProvisionActionType.TURN_ON and action.server_id is not None:
            s = self.servers[action.server_id]
            if s.state != ServerState.OFF:
                skip_reason = "not_off"
            elif s.in_cooldown(self.now, CFG.cooldown_sec):
                skip_reason = "cooldown"
            elif not turn_on_necessary and not bypass:
                skip_reason = "overload_not_sustained"
            else:
                necessary_now = self._was_turn_on_necessary_audit(snapshot)
                # *** یادداشت شفافیت (نه تغییر رفتار متریک): _was_turn_on_necessary_audit
                # از همان _any_service_capacity_starved(snapshot) استفاده می‌کند که
                # خودِ turn_on_necessary هم از آن استفاده کرده - یعنی وقتی دلیل
                # TURN_ON صرفاً capacity_starved بوده (نه sustained-overload)، این
                # ممیزی دارد همان واقعیت را روی همان snapshot دوباره می‌خواند، نه
                # یک چک واقعاً مستقل. این فی‌نفسه غلط نیست (starvation یک واقعیت
                # ساختاری عینی است) اما به این معنی است که این مسیر خاص هرگز
                # نمی‌تواند "نادرست" ثبت شود - برای شفافیت گزارش، این حالت جدا
                # لاگ می‌شود تا در تحلیل decision_correctness قابل تفکیک از
                # TURN_ONهای واقعاً مبتنی‌بر overload لحظه‌ای باشد.
                via_capacity_starved_only = (
                    self._any_service_capacity_starved(snapshot)
                    and not self._any_active_server_sustained_overloaded())
                self._start_server_boot(action.server_id)
                self.metrics.record_scale_action("TURN_ON")
                applied = True
                self.metrics.record_decision_correctness("TURN_ON", necessary_now)
                self._last_tick_applied_actions["provision"] = action
        elif action.action == ProvisionActionType.TURN_OFF and action.server_id is not None:
            s = self.servers[action.server_id]
            n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
            turn_off_necessary = self._was_turn_off_necessary(action.server_id)
            if s.state != ServerState.ACTIVE:
                skip_reason = "not_active"
            elif not turn_off_necessary and not bypass:
                skip_reason = "low_util_not_sustained"
            elif n_active <= 1:
                skip_reason = "last_active_server"
            elif s.in_cooldown(self.now, CFG.cooldown_sec):
                skip_reason = "cooldown"
            elif (not s.is_emergency_boot) and (self.now - s.last_transition_time) < CFG.min_active_duration_sec:
                skip_reason = "min_active_duration"
            else: 
                if self._start_server_drain(action.server_id):
                    self.metrics.record_scale_action("TURN_OFF")
                    applied = True
                    self.metrics.record_decision_correctness(
                        "TURN_OFF", self._was_turn_off_necessary_audit(action.server_id, snapshot))
                    self._last_tick_applied_actions["provision"] = action
                else:
                    skip_reason = "migration_incomplete"

        # *** رفع BUG-G: قبلاً هر دو حالت «الگوریتم اصلاً TURN_ON/OFF لازم
        # را پیشنهاد نداد» و «الگوریتم پیشنهاد داد ولی یک گیت سیستمی
        # (cooldown/migration ناقص/...) جلویش را گرفت» هر دو یکسان
        # missed_opportunity حساب می‌شدند - یعنی این معیار نمی‌توانست ضعف
        # واقعی الگوریتم را از یک محدودیت سیستمی موقت تفکیک کند. حالا حالت
        # دوم (پیشنهاد داده شد ولی مسدود شد) جدا در «blocked» ثبت می‌شود.
        proposed_turn_on = action.action == ProvisionActionType.TURN_ON and action.server_id is not None
        proposed_turn_off = action.action == ProvisionActionType.TURN_OFF and action.server_id is not None

        if turn_on_necessary and not applied:
            if proposed_turn_on:
                self.metrics.record_blocked_opportunity("TURN_ON")
            else:
                self.metrics.record_missed_opportunity("TURN_ON")
        if turn_off_opportunity and not (action.action == ProvisionActionType.TURN_OFF and applied):
            if proposed_turn_off:
                self.metrics.record_blocked_opportunity("TURN_OFF")
            else:
                self.metrics.record_missed_opportunity("TURN_OFF")

        self._log("provision_decision", action=action.action.name, server_id=action.server_id,
                  applied=applied, skip_reason=skip_reason,
                  necessary_turn_on=turn_on_necessary, turn_off_opportunity=turn_off_opportunity,
                  bypassed_sustain_gate=bypass,
                  via_capacity_starved_only=(via_capacity_starved_only if applied and
                      action.action == ProvisionActionType.TURN_ON else None))

    def _any_active_server_sustained_overloaded(self) -> bool:
        for sid, since in self._high_util_since.items():
            if since is not None and (self.now - since) >= CFG.sustain_high_sec:
                return True
        return False

    def _any_service_capacity_starved(self, snapshot: dict) -> bool:

        for svc_id in CFG.active_services:
            if not self._was_scale_up_necessary(svc_id, snapshot):
                continue
            cpu = CFG.services_info[svc_id]["resource_mips"]
            # *** رفع باگ: مرکز ثقل واقعی تقاضای همین سرویس (اگر موجود
            # باشد) به can_host پاس داده می‌شود تا بررسی SLA به‌جای
            # بدترین‌حالت همیشگی، از موقعیت واقعی ترافیک استفاده کند
            # (نگاه کنید common/models.py:Server.can_host).
            centroid = snapshot["services"][svc_id].get("demand_centroid")
            bts_lat, bts_long = centroid if centroid else (None, None)
            if not any(s.state in (ServerState.ACTIVE, ServerState.BOOTING)
                       and s.can_host(svc_id, cpu, bts_lat=bts_lat, bts_long=bts_long)
                       for s in self.servers.values()):
                return True
        return False
    def _any_active_server_sustained_underloaded(self) -> bool:
        n_active = sum(1 for s in self.servers.values() if s.state == ServerState.ACTIVE)
        if n_active <= 1:
            return False
        for sid, since in self._low_util_since.items():
            if since is not None and (self.now - since) >= CFG.sustain_low_sec:
                return True
        return False

    def _was_turn_off_necessary(self, server_id: int) -> bool:
        since = self._low_util_since.get(server_id)
        return since is not None and (self.now - since) >= CFG.sustain_low_sec
    def _was_turn_on_necessary_audit(self, snapshot: dict) -> bool:
        overloaded_now = any(
            snapshot["servers"][sid]["utilization"] > CFG.util_scale_up_threshold
            for sid, s in self.servers.items() if s.state == ServerState.ACTIVE
        )
        return overloaded_now or self._any_service_capacity_starved(snapshot)

    def _was_turn_off_necessary_audit(self, server_id: int, snapshot: dict) -> bool:
        return snapshot["servers"][server_id]["utilization"] < CFG.util_scale_down_threshold
    def _was_scale_up_necessary(self, svc_id: int, snapshot: dict) -> bool: 
        sv = snapshot["services"][svc_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
        return (occ_ratio > CFG.decision_audit_scale_up_occ_threshold
                or sv.get("rejection_rate_rolling", sv["rejection_rate"]) > 0.0
                or sv["deadline_violation_rate"] > 0.0)
    def _was_scale_down_necessary(self, svc_id: int, snapshot: dict) -> bool:
        sv = snapshot["services"][svc_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0 
        return occ_ratio < CFG.decision_audit_scale_down_occ_threshold and sv["n_ready_replicas"] > 1
    
    def _apply_scale_decision(self, svc_id: int, decision: ScaleAction, snapshot: dict):
        self._last_tick_decisions["scale"][svc_id] = decision
            
        applied = False
        skip_reason = None
        necessary_up = self._was_scale_up_necessary(svc_id, snapshot)
        necessary_down = self._was_scale_down_necessary(svc_id, snapshot)

        if decision == ScaleAction.NO_CHANGE:
            pass
        elif (self.now - self._service_last_scale_time[svc_id]) < CFG.cooldown_sec: 
            skip_reason = "cooldown"
        elif decision == ScaleAction.SCALE_UP:
            target = self.algorithm.select_placement_server(svc_id, self.servers)
            if target is not None:
                placed = self._place_replica(target, svc_id)
                if placed is not None:
                    self.metrics.record_scale_action("SCALE_UP")
                    self._service_last_scale_time[svc_id] = self.now
                    self._service_last_scale_up_time[svc_id] = self.now
                    applied = True
                    self.metrics.record_decision_correctness("SCALE_UP", necessary_up)
                    self._last_tick_applied_actions["scale"][svc_id] = decision
                else:
                    skip_reason = "placement_failed"
            else:
                skip_reason = "no_target_server"
        elif decision == ScaleAction.SCALE_DOWN: 
            ready = [r for r in self.replicas_by_service.get(svc_id, [])
                     if r.state == ReplicaState.READY]
            if len(ready) > 1:
                mature = [r for r in ready
                          if (self.now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
                if not mature:
                    skip_reason = "no_mature_replica"
                else:
                    victim = self.algorithm.select_scale_down_victim(svc_id, mature, self.servers, self.now)
                    self._start_replica_drain(victim)
                    self.metrics.record_scale_action("SCALE_DOWN")
                    self._service_last_scale_time[svc_id] = self.now
                    applied = True
                    self.metrics.record_decision_correctness("SCALE_DOWN", necessary_down)
                    self._last_tick_applied_actions["scale"][svc_id] = decision
            else:
                skip_reason = "only_one_replica_left"
 
        # *** رفع باگ (تکمیل BUG-G برای مسیر scale): همان تفکیکی که
        # _apply_provisioning بین «الگوریتم اصلاً پیشنهاد نداد» (missed) و
        # «الگوریتم دقیقاً همین اکشن را پیشنهاد داد ولی یک گیت سیستمی مثل
        # cooldown/no_target_server/no_mature_replica جلویش را گرفت»
        # (blocked) قائل می‌شود، اینجا هم اعمال می‌شود - قبلاً هر دو حالت
        # یکسان missed_opportunity حساب می‌شدند و decision_correctness
        # نمی‌توانست ضعف واقعی الگوریتم را از یک محدودیت سیستمی موقت تفکیک
        # کند.
        proposed_scale_up = decision == ScaleAction.SCALE_UP
        proposed_scale_down = decision == ScaleAction.SCALE_DOWN

        if necessary_up and not (decision == ScaleAction.SCALE_UP and applied):
            if proposed_scale_up:
                self.metrics.record_blocked_opportunity("SCALE_UP")
            else:
                self.metrics.record_missed_opportunity("SCALE_UP")
        if necessary_down and not (decision == ScaleAction.SCALE_DOWN and applied):
            if proposed_scale_down:
                self.metrics.record_blocked_opportunity("SCALE_DOWN")
            else:
                self.metrics.record_missed_opportunity("SCALE_DOWN")
 
        self._log("scale_decision", service_id=svc_id, decision=decision.name,
                  applied=applied, skip_reason=skip_reason,
                  necessary_scale_up=necessary_up, necessary_scale_down=necessary_down)

    def _update_sustain_tracking(self, snapshot: dict):
        for sid, s in self.servers.items():
            if s.state != ServerState.ACTIVE:
                self._low_util_since[sid] = None
                self._high_util_since[sid] = None
                continue
            util = snapshot["servers"][sid]["utilization"]
            if util < CFG.util_scale_down_threshold:
                if self._low_util_since[sid] is None:
                    self._low_util_since[sid] = self.now
            else:
                self._low_util_since[sid] = None

            if util > CFG.util_scale_up_threshold:
                if self._high_util_since[sid] is None:
                    self._high_util_since[sid] = self.now
            else:
                self._high_util_since[sid] = None

    def _annotate_provisioning_necessity(self, snapshot: dict):
        """(رفع باگ: هم‌راستایی action mask آموزش/inference برای PPO)

        algorithms/ppo/env.py (مسیر آموزش) مستقیماً به خودِ self.engine
        دسترسی دارد و می‌تواند _any_active_server_sustained_overloaded /
        _any_service_capacity_starved / _was_turn_off_necessary را زنده
        صدا بزند تا mask دقیقاً با گیت واقعی _apply_provisioning یکی باشد.
        اما algorithms/ppo/ppo_algorithm.py (مسیر inference/k8s واقعی) فقط
        (servers, snapshot, now) می‌گیرد و اصلاً به نمونه‌ی engine دسترسی
        ندارد - قبلاً همین باعث می‌شد mask زمان inference از mask زمان
        آموزش عقب بماند (فقط state+cooldown را چک می‌کرد، نه واقعاً لازم
        بودن) و مدل رفتاری متفاوت از چیزی که آموزش دیده ببیند.

        راه‌حل: به‌جای تکرار منطق sustain-tracking در ppo_algorithm.py (که
        خطر واگرایی بعدی دارد)، همان سیگنال‌های تازه‌محاسبه‌شده مستقیماً در
        خودِ snapshot ذخیره می‌شوند - یک منبع واحد حقیقت، هم برای مسیر
        آموزش (اگر بخواهد) هم برای هر AlgorithmBase دیگری."""
        snapshot["global"]["turn_on_necessary"] = (
            self._any_active_server_sustained_overloaded()
            or self._any_service_capacity_starved(snapshot))
        for sid in self.servers:
            snapshot["servers"][sid]["turn_off_necessary"] = self._was_turn_off_necessary(sid)

    def _handle_decision_tick(self, external_actions: dict | None = None) -> dict: 
        self.metrics.record_snapshot(self.now, self.servers)
        snapshot = self._build_metrics_snapshot()
        self._update_sustain_tracking(snapshot)
        self._annotate_provisioning_necessity(snapshot)
        self._last_tick_decisions = {"provision": None, "scale": {}}
        self._last_tick_applied_actions = {"provision": None, "scale": {}}

        if external_actions is not None:
            self._apply_provisioning(external_actions["provision"], snapshot)
            for svc_id, decision in external_actions["scale"].items():
                self._apply_scale_decision(svc_id, decision, snapshot)
        else:
            action = self.algorithm.provision_decision(self.servers, snapshot, self.now)
            self._apply_provisioning(action, snapshot)
            for svc_id in CFG.active_services:
                decision = self.algorithm.scale_decision(svc_id, snapshot)
                self._apply_scale_decision(svc_id, decision, snapshot)

        for sid, s in self.servers.items():
            self._util_at_window_start[sid] = s.cumulative_busy_cpu_seconds
        self._util_window_start_time = self.now

        self._tick_total.clear()
        self._tick_rejected.clear()
        self._tick_violated.clear()
        self._tick_response_times.clear()
        self._tick_lat_sum.clear()
        self._tick_lon_sum.clear()
        self._tick_proximity_violated.clear()
        return snapshot
 
    def run(self) -> dict:
        self.prime()
        while True:
            _, done = self.step()
            if done:
                break
        return self.metrics.finalize(self.servers)
 
    def prime(self): 
        for row in self.events_df.itertuples(index=False):
            self._push(float(row.global_start_sec), EventType.REQUEST_ARRIVAL, row)
        start_time = float(self.events_df.global_start_sec.min()) if len(self.events_df) else 0.0
        max_time = float(self.events_df.global_start_sec.max()) if len(self.events_df) else 0.0
        self.now = start_time
            
        self._energy_last_update = {sid: start_time for sid in self.servers} 
        self._util_window_start_time = start_time 
        
        self._initial_placement()
        self._push(start_time, EventType.DECISION_TICK) 
        self._cutoff = (max_time + 2 * CFG.dispatch_overhead_ms / 1000.0
                         + CFG.decision_interval_sec + CFG.server_drain_grace_sec)

    def step(self, external_actions: dict | None = None): 
        while self._heap:
            ev = heapq.heappop(self._heap)
            if ev.time > self._cutoff:
                return None, True
            self._advance_energy_to(ev.time)
            self.now = ev.time

            if ev.type == EventType.REQUEST_ARRIVAL:
                self._handle_arrival(ev.payload)
            elif ev.type == EventType.REQUEST_ROUTED:
                self._handle_routed(ev.payload)
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
            elif ev.type == EventType.ENERGY_RESYNC:
                pass   
        return None, True