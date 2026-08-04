"""
k8s_adapter/k8s_client.py

لایه‌ی نازک روی kubernetes Python client برای عملیات واقعی فاز ۳.

*** پیش‌نیازهای یک‌باره روی کلاستر شما (قبل از استفاده از این ماژول):
    ۱) هر ۱۰ worker node را طبق SERVER_INFO لیبل بزنید تا nodeSelector کار کند:
           kubectl label node <نام‌نود-متناظر-با-192.168.1.11> edge-server-id=1
           kubectl label node <نام‌نود-متناظر-با-192.168.1.12> edge-server-id=2
           ... (تا سرور ۱۰ -> 192.168.1.20)
       (خودِ این پروژه نمی‌داند نام واقعی نودهای شما در `kubectl get nodes`
       چیست؛ فقط IP را در common/config.py:SERVER_INFO می‌دانیم - این نگاشت
       IP<->نام‌نود را باید یک‌بار خودتان با `kubectl get nodes -o wide` پیدا
       و لیبل‌گذاری کنید.)
    ۲) namespace بسازید: kubectl create namespace edge-rl
    ۳) image سرویس (k8s_adapter/worker_service/) را build/push کنید و
       نامش را در common/config.py یا آرگومان WORKER_IMAGE اینجا بدهید.
    ۴) روی ماشین ۱۹۲.۱۶۸.۱.۳۰ (جایی که این کد اجرا می‌شود)، `~/.kube/config`
       باید به کلاستر شما دسترسی داشته باشد (همان kubeconfig که kubectl
       استفاده می‌کند).
"""

from __future__ import annotations
import time
from typing import Optional
import time as _time

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
except ImportError as e:
    raise ImportError(
        "کتابخانه‌ی kubernetes نصب نیست. اجرا کنید: pip install kubernetes"
    ) from e

from common.config import CFG

NAMESPACE = "edge-rl"
WORKER_IMAGE = "192.168.1.30:5000/edge-worker:latest"
NODE_LABEL_KEY = "edge-server-id"

# *** تبدیل واحد انتزاعی cpu_demand این پروژه به میلی‌سی‌پی‌یوی واقعی K8s.
# چون cpu_demand مقیاس دلخواه ماست (نه core واقعی)، این ضریب را با توجه به
# سخت‌افزار واقعی worker نودهایتان کالیبره کنید (پیش‌فرض: هر واحد = 50m).
CPU_UNIT_TO_MILLICPU = 50


def _load_kube_config():
    try:
        config.load_kube_config()  # از ~/.kube/config (حالت معمول خارج از کلاستر)
    except Exception:
        config.load_incluster_config()  # اگر خودِ کد داخل یک پاد در کلاستر اجرا شود


_load_kube_config()
_apps_v1 = client.AppsV1Api()
_core_v1 = client.CoreV1Api()


def _deployment_name(service_id: int, server_id: int) -> str:
    return f"svc{service_id}-srv{server_id}"


def worker_port(service_id: int) -> int:
    """
    *** پورت اختصاصیِ هر سرویس (نه هر پاد). چون از این پس با hostNetwork=True
    اجرا می‌کنیم، اگر چند سرویس روی یک نود زمان‌بندی بشن، همه روی IP واقعیِ
    همان نود گوش می‌دهند و اگر همه پورت ثابت 8000 داشتند تداخل پیش می‌آمد؛
    پس هر service_id پورت خودش را می‌گیرد (دقیقاً همان الگویی که در
    پروژه‌ی قبلی‌تان با SERVICE_PORT = SERVICE_ID + 8000 کار می‌کرد).
    """
    return 8000 + service_id


# ---------------------------------------------------------------------------
# Deployment (رپلیکا) - ایجاد/حذف/وضعیت
# ---------------------------------------------------------------------------

def build_deployment_manifest(service_id: int, server_id: int) -> client.V1Deployment:
    svc = CFG.services_info[service_id]
    name = _deployment_name(service_id, server_id)
    cpu_millicpu = svc["cpu_demand"] * CPU_UNIT_TO_MILLICPU
    port = worker_port(service_id)

    container = client.V1Container(
        name="worker",
        image=WORKER_IMAGE,
        ports=[client.V1ContainerPort(container_port=port, host_port=port)],
        env=[
            client.V1EnvVar(name="EXEC_TIME_SEC", value=str(svc["exec_time"])),
            client.V1EnvVar(name="SERVICE_ID", value=str(service_id)),
            client.V1EnvVar(name="SERVER_ID", value=str(server_id)),
            client.V1EnvVar(name="SERVICE_PORT", value=str(port)),
        ],
        resources=client.V1ResourceRequirements(
            requests={"cpu": f"{cpu_millicpu}m", "memory": svc["memory"]},
            limits={"cpu": f"{cpu_millicpu}m", "memory": svc["memory"]},
        ),
        readiness_probe=client.V1Probe(
            http_get=client.V1HTTPGetAction(path="/healthz", port=port),
            initial_delay_seconds=1, period_seconds=2, failure_threshold=3,
        ),
    )

    labels = {"app": "edge-worker", "service_id": str(service_id), "server_id": str(server_id)}
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=client.V1PodSpec(
            containers=[container],
            node_selector={NODE_LABEL_KEY: str(server_id)},
            # *** کلید اصلی رفع مشکل timeout: با hostNetwork=True، پاد روی
            # IP واقعی نود (192.168.1.x) گوش می‌دهد و pod.status.pod_ip هم
            # همان IP واقعی نود را برمی‌گرداند - یعنی از ماشین base مستقیماً
            # reachable است (برخلاف IP overlay شبکه‌ی CNI که قبلاً برمی‌گشت).
            host_network=True,
            dns_policy="ClusterFirstWithHostNet",
        ),
    )

    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels=labels),
        template=template,
    )

    return client.V1Deployment(
        metadata=client.V1ObjectMeta(name=name, namespace=NAMESPACE, labels=labels),
        spec=spec,
    )


def create_deployment(service_id: int, server_id: int):
    """معادل _place_replica در simulator/engine.py؛ pod-create واقعی."""
    manifest = build_deployment_manifest(service_id, server_id)
    try: 
        _call_with_retry(_apps_v1.create_namespaced_deployment,namespace=NAMESPACE, body=manifest)
    except ApiException as e:
        if e.status == 409:  # از قبل وجود دارد
            return
        raise


def delete_deployment(service_id: int, server_id: int):
    """معادل _handle_replica_terminated؛ pod-delete واقعی."""
    name = _deployment_name(service_id, server_id)
    try: 
        _call_with_retry(_apps_v1.delete_namespaced_deployment,name=name, namespace=NAMESPACE)
    except ApiException as e:
        if e.status == 404:
            return
        raise

 
 
def is_deployment_ready(service_id: int, server_id: int) -> bool:
    name = _deployment_name(service_id, server_id)
    try:
        dep = _call_with_retry(_apps_v1.read_namespaced_deployment_status,
                                name=name, namespace=NAMESPACE)
    except ApiException as e:
        if e.status == 404:
            return False 
        return False
    return (dep.status.ready_replicas or 0) >= 1


def get_pod_ip(service_id: int, server_id: int) -> Optional[str]:
    label_selector = f"service_id={service_id},server_id={server_id}"
    pods = _core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector)
    for pod in pods.items:
        if pod.status.phase == "Running" and pod.status.pod_ip:
            return pod.status.pod_ip
    return None


def list_all_deployments() -> list[dict]:
    """برای همگام‌سازی وضعیت اولیه‌ی Redis با آنچه واقعاً روی کلاستر هست."""
    deployments = _apps_v1.list_namespaced_deployment(
        namespace=NAMESPACE, label_selector="app=edge-worker")
    out = []
    for d in deployments.items:
        out.append({
            "service_id": int(d.metadata.labels["service_id"]),
            "server_id": int(d.metadata.labels["server_id"]),
            "ready": (d.status.ready_replicas or 0) >= 1,
        })
    return out
 

def _call_with_retry(func, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """
    *** لایه‌ی مقاومت در برابر خطاهای گذرای API Server (۵۰۰/۵۰۳/۴۲۹ - از
    جمله «context deadline exceeded» که از etcd کند می‌آید). بدون این، یک
    خطای گذرا کل کار (create/poll) را برای همیشه می‌کشد، چون بالادست
    (asyncio task در realtime_dispatcher.py) هیچ retry ندارد.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except ApiException as e:
            last_exc = e
            if e.status not in (500, 503, 429):
                raise  # خطای غیرگذرا (مثلاً 404/401) - retry بی‌فایده است
            _time.sleep(base_delay * (2 ** attempt))
    raise last_exc
# ---------------------------------------------------------------------------
# Node (سرور) - cordon/uncordon (طبق پاسخ شما: بدون خاموشی فیزیکی واقعی)
# ---------------------------------------------------------------------------

def _get_node_name(server_id: int) -> str:
    nodes = _core_v1.list_node(label_selector=f"{NODE_LABEL_KEY}={server_id}")
    if not nodes.items:
        raise RuntimeError(
            f"هیچ نودی با لیبل {NODE_LABEL_KEY}={server_id} پیدا نشد. "
            f"اول طبق دستور بالای فایل نودها را لیبل بزنید."
        )
    return nodes.items[0].metadata.name


def cordon_node(server_id: int):
    """معادل server -> OFF/DRAINING: از این پس پاد جدید رویش schedule نمی‌شود."""
    node_name = _get_node_name(server_id)
    body = {"spec": {"unschedulable": True}}
    _core_v1.patch_node(node_name, body)


def uncordon_node(server_id: int):
    """معادل server -> ACTIVE: دوباره قابل schedule می‌شود."""
    node_name = _get_node_name(server_id)
    body = {"spec": {"unschedulable": False}}
    _core_v1.patch_node(node_name, body)


def is_node_schedulable(server_id: int) -> bool:
    node_name = _get_node_name(server_id)
    node = _core_v1.read_node(node_name)
    return not bool(node.spec.unschedulable)