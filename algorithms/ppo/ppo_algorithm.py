"""
algorithms/ppo/ppo_algorithm.py
 
"""

from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.models import Server, ServerState
from common.state_builder import build_state_vector
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep
from algorithms.greedy.greedy_algorithm import GreedyAlgorithm

from common.geo import haversine_km, network_delay_ms
_SERVICE_IDS = sorted(CFG.services_info.keys())
_SERVER_IDS = sorted(CFG.server_info.keys())
_SCALE_MAP = {0: ScaleAction.NO_CHANGE, 1: ScaleAction.SCALE_UP, 2: ScaleAction.SCALE_DOWN}
_PROVISION_MAP = {0: ProvisionActionType.NO_CHANGE, 1: ProvisionActionType.TURN_ON,
                   2: ProvisionActionType.TURN_OFF}


class PPOAlgorithm(AlgorithmBase):
    name = "ppo"

    # *** جدید: برخلاف Greedy/HPA/VOILA، تصمیمات provisioning (TURN_ON/
    # TURN_OFF) این الگوریتم لازم نیست منتظر گیت sustain-tracking مشترک
    # (SUSTAIN_HIGH_SEC/SUSTAIN_LOW_SEC در simulator/engine.py و
    # k8s_adapter/realtime_dispatcher.py:_apply_provisioning) بمانند. دلیل:
    # action mask (algorithms/ppo/env.py:compute_action_masks) از قبل فقط
    # امکان‌پذیری فیزیکی را چک می‌کند و به PPO اجازه می‌دهد هر زمان TURN_ON/
    # TURN_OFF را *انتخاب* کند؛ بدون این پرچم، آن انتخاب توسط
    # _apply_provisioning همیشه به NO_CHANGE تنزل پیدا می‌کرد (چون سیگنال
    # necessity هنوز sustained نشده) و PPO هرگز فرصت واقعی نداشت رفتار
    # پیش‌بینانه (anticipatory) را در عمل امتحان و از طریق reward واقعی
    # (تعادل هزینه‌ی انرژی/اکشن در برابر کاهش زمان پاسخ و نقض deadline)
    # یاد بگیرد. توجه: cooldown و min_active_duration صرف‌نظر از این پرچم
    # همچنان اعمال می‌شوند (قیود عملیاتی سخت، نه بخشی از این آستانه‌ی
    # reactive)؛ ممیزی decision_correctness هم دست‌نخورده می‌ماند - این
    # پرچم فقط تعیین می‌کند اکشن اعمال می‌شود یا نه، نه اینکه "درست" شمرده
    # شود.
    bypass_sustain_gate: bool = True
 
    def __init__(self, model_path, deterministic=True, latency_aware_routing=False , use_solver_placement = True, placement_weights=None): 
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as e:
            raise ImportError(
                "sb3-contrib نصب نیست. اجرا کنید: pip install -r requirements.txt"
            ) from e
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic
        self._cached_tick_key: Optional[float] = None
        self._cached_scale: Dict[int, ScaleAction] = {}
        self._cached_provision = ProvisionAction(ProvisionActionType.NO_CHANGE) 
        self._last_snapshot: Optional[dict] = None 
        self._helper = GreedyAlgorithm()
        self.latency_aware_routing = latency_aware_routing
 
        self.use_solver_placement = use_solver_placement
        self._solver_selected_servers: Optional[List[int]] = None
 
        self._placement_weights = placement_weights or {"w_count": 1.0, "w_energy": 1.0, "w_distance": 1.0} 
        self._infer_svc_last_scale: dict = {sid: -1e18 for sid in sorted(CFG.services_info.keys())}
      
    # ------------------------------------------------------------------
    def initial_placement(self, servers: Dict[int, Server], active_bts):
        if not self.use_solver_placement:
            return super().initial_placement(servers, active_bts)

        if self._solver_selected_servers is None:
            self._solver_selected_servers = self._solve_from_training_data(servers, active_bts)

        return self._ensure_sufficient_capacity(servers, list(self._solver_selected_servers))


    def _solve_from_training_data(self, servers, active_bts_fallback):
        from algorithms.ppo.optimal_placement import aggregate_training_demand, solve_optimal_server_selection
        try:
            from data.loader import load_train
            train_events = load_train()
            demand_points = aggregate_training_demand(train_events)
            selected = solve_optimal_server_selection(servers, demand_points, **self._placement_weights)
            if selected:
                print(f"[PPO] جای‌گذاری اولیه‌ی چندهدفه حل شد: {len(selected)} سرور، "
                    f"سرورها: {sorted(selected)}")
                return selected
            print("[PPO] solver جواب قابل‌قبول پیدا نکرد؛ fallback به پوشش حریصانه‌ی مشترک.")
        except Exception as e:
            print(f"[PPO] حل ILP شکست خورد ({e}); fallback به پوشش حریصانه‌ی مشترک.")
        return super(PPOAlgorithm, self).initial_placement(servers, active_bts_fallback)

    @staticmethod
    def _ensure_sufficient_capacity(servers: Dict[int, Server], selected: List[int]) -> List[int]: 
        total_cpu_needed = sum(s["resource_mips"] for s in CFG.services_info.values())
        if sum(servers[sid].capacity for sid in selected) >= total_cpu_needed:
            return selected
        remaining = [sid for sid in servers if sid not in selected]
        remaining.sort(key=lambda sid: min(
            haversine_km(servers[sid].lat, servers[sid].long, servers[s2].lat, servers[s2].long)
            for s2 in selected) if selected else 0)
        while sum(servers[sid].capacity for sid in selected) < total_cpu_needed and remaining:
            selected.append(remaining.pop(0))
        return selected
    # ------------------------------------------------------------------
    def _predict_and_cache(self, servers: Dict[int, Server], metrics_snapshot: dict, now: float): 
        self._last_snapshot = metrics_snapshot
        if self._cached_tick_key == now:
            return 
        obs = build_state_vector(metrics_snapshot, servers)
        action_masks = self._build_action_masks(servers, metrics_snapshot, now=now)
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=self.deterministic)

        self._cached_scale = {sid: _SCALE_MAP[int(action[i])] for i, sid in enumerate(_SERVICE_IDS)}
 
        for _sid, _act in self._cached_scale.items():
            if _act != ScaleAction.NO_CHANGE:
                self._infer_svc_last_scale[_sid] = now
  
        non_noop = []
        for j, sid in enumerate(_SERVER_IDS):
            ptype = _PROVISION_MAP[int(action[len(_SERVICE_IDS) + j])]
            if ptype != ProvisionActionType.NO_CHANGE:
                non_noop.append((sid, ptype))
        provision = ProvisionAction(ProvisionActionType.NO_CHANGE)
        if non_noop:
            turn_ons = sorted((sid, pt) for sid, pt in non_noop if pt == ProvisionActionType.TURN_ON)
            turn_offs = sorted((sid, pt) for sid, pt in non_noop if pt == ProvisionActionType.TURN_OFF)
            chosen_sid, chosen_ptype = (turn_ons or turn_offs)[0]
            provision = ProvisionAction(chosen_ptype, chosen_sid)
        self._cached_provision = provision
        self._cached_tick_key = now

    def _build_action_masks(self, servers: Dict[int, Server], snapshot: dict, now: float | None = None): 
        import numpy as np
        masks = []
        t = now if now is not None else 0.0
        for sid in _SERVICE_IDS:
            sv = snapshot["services"][sid]
            cpu = CFG.services_info[sid]["resource_mips"] 
            in_svc_cooldown = snapshot["services"][sid].get("scale_cooldown_active", False)
            # *** رفع باگ: can_host بدون demand_centroid صدا زده می‌شد، یعنی
            # mask می‌توانست SCALE_UP یک سرویس را غیرمجاز اعلام کند (بدترین‌
            # حالت fail) حتی وقتی select_placement_server (چند خط پایین‌تر
            # در همین فایل) با همان centroid واقعاً یک سرور پیدا می‌کرد.
            _centroid = sv.get("demand_centroid")
            _bts_lat, _bts_long = _centroid if _centroid else (None, None)
            can_up = (not in_svc_cooldown) and any(
                s.state == ServerState.ACTIVE
                and s.can_host(sid, cpu, bts_lat=_bts_lat, bts_long=_bts_long)
                for s in servers.values())
            n_mature = sv.get("n_mature_ready_replicas", sv["n_ready_replicas"])
            can_down = (not in_svc_cooldown) and sv["n_ready_replicas"] > 1 and n_mature > 0
            masks.extend([True, can_up, can_down])
        # *** رفع بازبینی (برگرداندن BUG-A، هم‌راستا با algorithms/ppo/env.py):
        # نسخه‌ی قبلی can_on/can_off را با turn_on_necessary/turn_off_necessary
        # (تولیدشده در simulator/engine.py:_annotate_provisioning_necessity و
        # k8s_adapter/realtime_dispatcher.py:_annotate_provisioning_necessity)
        # هم گیت می‌کرد. نیت آن («mask دقیقاً با گیت واقعی موتور یکی باشد»)
        # درست بود، ولی چون در inference هم مدل از قبل با ماسکِ necessity-gated
        # آموزش دیده، این گیت را همینجا هم نگه داشتن فقط یک محدودیت اضافه‌ی
        # بی‌فایده روی مدلی است که دیگر اصلاً یاد نگرفته زودتر از sustain
        # عمل کند. توجیه اصلی («بدون آن PPO سیگنال reward نویزی می‌گیرد») هم
        # نادرست بود: n_actions_applied (نگاه کنید algorithms/ppo/env.py:step)
        # فقط از شمارنده‌های num_turn_on/off *پس از* اعمال واقعی ساخته
        # می‌شود، یعنی یک TURN_ON ردشده با skip_reason="overload_not_sustained"
        # از دید reward/observation دقیقاً معادل NO_CHANGE است. پس mask اینجا
        # هم باید فقط امکان‌پذیری *فیزیکی* را چک کند - نه توافق با Greedy -
        # تا فضای تصمیم inference دقیقاً با فضای تصمیمی که مدل با آن آموزش
        # دیده (بعد از رفع همین باگ در env.py) یکی بماند.
        # توابع _annotate_provisioning_necessity و فیلدهای
        # turn_on_necessary/turn_off_necessary در snapshot حذف نشده‌اند -
        # همچنان برای گزارش/ممیزی decision_correctness در
        # simulator/engine.py:_apply_provisioning استفاده می‌شوند؛ فقط دیگر
        # در این mask اعمال نمی‌شوند.
        for sid in _SERVER_IDS:
            st = snapshot["servers"][sid]["state"]
            s = servers[sid] 

            can_on = (st == ServerState.OFF
                    and not snapshot["servers"][sid]["provision_cooldown_active"])
            can_off = (st == ServerState.ACTIVE
                    and not snapshot["servers"][sid]["provision_cooldown_active"]
                    and not snapshot["servers"][sid]["is_last_active_server"]
                    and snapshot["servers"][sid]["min_active_duration_met"])
            masks.extend([True, can_on, can_off])
        return np.array(masks, dtype=bool)

    # ------------------------------------------------------------------
    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        return self._cached_scale.get(service_id, ScaleAction.NO_CHANGE)

    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        self._predict_and_cache(servers, metrics_snapshot, now)
        return self._cached_provision

    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]: 
        cpu = CFG.services_info[service_id]["resource_mips"]

        centroid = None
        if self._last_snapshot is not None:
            centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
        if centroid is None:
            active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
            clat = sum(s.lat for s in active) / len(active)
            clon = sum(s.long for s in active) / len(active)
            centroid = (clat, clon)

        # *** رفع باگ (fairness/SLA feasibility): centroid از قبل محاسبه
        # می‌شد ولی فقط برای مرتب‌سازی فاصله استفاده می‌شد، نه برای خودِ چک
        # can_host - یعنی فیلتر SLA feasibility همیشه بدون مختصات (بدترین
        # حالت ۴ گوشه‌ی نقشه) اجرا می‌شد، درست مثل Greedy/HPA و برخلاف
        # VOILA. الان centroid همان‌جا که محاسبه شده به can_host هم پاس
        # داده می‌شود.
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE
                      and s.can_host(service_id, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
        if not candidates:
            return None

        distances = {s.id: haversine_km(centroid[0], centroid[1], s.lat, s.long) for s in candidates}
        min_dist = min(distances.values())
        near_pool = [s for s in candidates if distances[s.id] <= min_dist + 5.0]
        return max(near_pool, key=lambda s: s.free_capacity()).id

    def migration_decision(self, draining_server: Server,
                            servers: Dict[int, Server]) -> List[MigrationStep]:
        return self._helper.migration_decision(draining_server, servers)

    def select_replica(self, request, candidate_replicas, servers, now, admit_fn=None, occupancy_fn=None):
        if not candidate_replicas:
            return None
        if not self.latency_aware_routing:
            return super().select_replica(request, candidate_replicas, servers, now,
                                           admit_fn=admit_fn, occupancy_fn=occupancy_fn)

        # *** فیکس: مثل base.select_replica، خواندن (occupancy_fn) از رزرو
        # واقعی (admit_fn) جدا شده تا در حالت realtime که admit_fn واقعاً یک
        # اسلات صف را در Redis رزرو می‌کند، روی همه‌ی کاندیدها هم‌زمان صدا
        # زده نشود (وگرنه فقط یکی استفاده و بقیه تا انقضای TTL نشت می‌کنند).
        # *** فیکس: r.available_at در مسیر realtime هرگز آپدیت نمی‌شود
        # (چون try_admit فقط در simulator صدا زده می‌شود)، پس همیشه ۰ می‌ماند
        # و "ترافیک" را کاملاً نادیده می‌گرفت. حالا تأخیر صف از روی همان
        # occupancy واقعی (که در realtime از Redis می‌آید) تخمین زده می‌شود:
        # occupancy فعلی × زمان اجرای هر درخواست ≈ زمان انتظار تخمینی.
        occupancy_fn = occupancy_fn or (lambda r: r.queue_occupancy(now))
        admit_fn = admit_fn or (lambda r: occupancy_fn(r) < r.queue_len)

        ranked = []
        for r in candidate_replicas:
            occ = occupancy_fn(r)
            if occ >= r.queue_len:
                continue  # از قبل معلوم است رد می‌شود؛ رزروش نکن
            server = servers[r.server_id]
            distance_km = haversine_km(request.bts_lat, request.bts_long, server.lat, server.long)
            delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
            rtt_sec = 2 * delay_ms / 1000.0

            est_wait_sec = occ * r.exec_time
            est_total_latency = rtt_sec + est_wait_sec + r.exec_time
            ranked.append((est_total_latency, r))

        # به ترتیب کمترین تأخیر تخمینی امتحان کن؛ رزروِ واقعی فقط روی
        # کاندیدایی انجام می‌شود که واقعاً انتخاب و برگردانده می‌شود
        ranked.sort(key=lambda pair: pair[0])
        for _, r in ranked:
            if admit_fn(r):
                return r
        return None