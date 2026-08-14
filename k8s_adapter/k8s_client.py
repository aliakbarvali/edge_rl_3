"""
k8s_adapter/k8s_client.py
"""

from __future__ import annotations
import time
from typing import Optional
import time as _time
from common.config import CFG, compute_exec_time_sec, compute_cold_start_window_sec, compute_cold_start_penalty_sec
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


def resource_mips_to_millicpu(resource_mips: int) -> int:
    from common.config import REFERENCE_MIPS_PER_CORE
    return round(resource_mips / REFERENCE_MIPS_PER_CORE * 1000)


def _load_kube_config():
    try:
        config.load_kube_config()   
    except Exception:
        config.load_incluster_config()  


_load_kube_config()
_apps_v1 = client.AppsV1Api()
_core_v1 = client.CoreV1Api()


def _deployment_name(service_id: int, server_id: int) -> str:
    return f"svc{service_id}-srv{server_id}"


def worker_port(service_id: int) -> int: 
    return 8000 + service_id


def build_deployment_manifest(service_id: int, server_id: int) -> client.V1Deployment:
    svc = CFG.services_info[service_id]
    name = _deployment_name(service_id, server_id)
    
    server_profile = CFG.server_profiles[CFG.server_info[server_id]["profile"]]
    cpu_millicpu = resource_mips_to_millicpu(svc["resource_mips"])
    exec_time_sec = compute_exec_time_sec(service_id, server_profile["mips_per_core"])

    # *** پچ (رفع باگ ۱ - cold-start penalty گم‌شده در مسیر k8s): پنجره/جریمه‌ی
    # cold-start هم مثل exec_time_sec همین‌جا (جایی که هم service_id هم
    # server_profile در دسترس‌اند) محاسبه و به‌عنوان env var به پاد پاس داده
    # می‌شود - نگاه کنید k8s_adapter/worker_service/app.py برای توضیح کامل و
    # اینکه چرا محاسبه اینجاست نه داخل خودِ پاد (image پاد فقط app.py دارد،
    # نه ماژول common/).
    cold_start_window_sec = compute_cold_start_window_sec(service_id, server_profile["mips_per_core"])
    cold_start_penalty_sec = compute_cold_start_penalty_sec(service_id, server_profile["mips_per_core"])

    port = worker_port(service_id)

    container = client.V1Container(
        name="worker",
        image=WORKER_IMAGE,
        ports=[client.V1ContainerPort(container_port=port, host_port=port)],
        env=[
            client.V1EnvVar(name="EXEC_TIME_SEC", value=str(exec_time_sec)),
            client.V1EnvVar(name="SERVICE_ID", value=str(service_id)),
            client.V1EnvVar(name="SERVER_ID", value=str(server_id)),
            client.V1EnvVar(name="SERVICE_PORT", value=str(port)),
            client.V1EnvVar(name="COLD_START_WINDOW_SEC", value=str(cold_start_window_sec)),
            client.V1EnvVar(name="COLD_START_PENALTY_SEC", value=str(cold_start_penalty_sec)),
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
    manifest = build_deployment_manifest(service_id, server_id)
    try: 
        _call_with_retry(_apps_v1.create_namespaced_deployment,namespace=NAMESPACE, body=manifest)
    except ApiException as e:
        if e.status == 409:
            return
        raise


def delete_deployment(service_id: int, server_id: int):
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
        raise
    return (dep.status.ready_replicas or 0) >= 1


def get_pod_ip(service_id: int, server_id: int) -> Optional[str]:
    label_selector = f"service_id={service_id},server_id={server_id}"
    pods = _core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector)
    for pod in pods.items:
        if pod.status.phase == "Running" and pod.status.pod_ip:
            return pod.status.pod_ip
    return None


def list_all_deployments() -> list[dict]:
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
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except ApiException as e:
            last_exc = e
            if e.status not in (500, 503, 429):
                raise  
            _time.sleep(base_delay * (2 ** attempt))
    raise last_exc
 

def _get_node_name(server_id: int) -> str:
    nodes = _core_v1.list_node(label_selector=f"{NODE_LABEL_KEY}={server_id}")
    if not nodes.items:
        raise RuntimeError(
            f"هیچ نودی با لیبل {NODE_LABEL_KEY}={server_id} پیدا نشد. "
            f"اول طبق دستور بالای فایل نودها را لیبل بزنید."
        )
    return nodes.items[0].metadata.name


def cordon_node(server_id: int):
    node_name = _get_node_name(server_id)
    body = {"spec": {"unschedulable": True}}
    _core_v1.patch_node(node_name, body)


def uncordon_node(server_id: int):
    node_name = _get_node_name(server_id)
    body = {"spec": {"unschedulable": False}}
    _core_v1.patch_node(node_name, body)


def is_node_schedulable(server_id: int) -> bool:
    node_name = _get_node_name(server_id)
    node = _core_v1.read_node(node_name)
    return not bool(node.spec.unschedulable)