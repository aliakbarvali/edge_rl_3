"""
k8s_adapter/smoke_test.py

قبل از اجرای کامل realtime_dispatcher.py، این تست رو اجرا کنید تا مطمئن
بشید اتصال Redis و Kubernetes درست کار می‌کنه و یک Deployment آزمایشی
واقعاً بالا میاد و پاسخ می‌ده. اگه اینجا خطا گرفتید، اول همینو دیباگ کنید
قبل از اجرای کامل روی کل روز داده.

اجرا:
    python3 -m k8s_adapter.smoke_test
"""

from __future__ import annotations
import sys
import time


def test_redis():
    print("۱) تست اتصال Redis ...")
    from k8s_adapter import redis_state
    try:
        ok = redis_state.ping()
        assert ok
        redis_state.set_server_state(999, "TEST")
        assert redis_state.get_server_state(999) == "TEST"
        redis_state._r.delete("edge:server:999:state")
        print("   ✓ Redis وصل شد و read/write کار می‌کند.")
        return True
    except Exception as e:
        print(f"   ✗ خطا در اتصال به Redis (192.168.1.30:6379): {e}")
        return False


def test_k8s_connection():
    print("۲) تست اتصال به Kubernetes API ...")
    try:
        from k8s_adapter import k8s_client
        nodes = k8s_client._core_v1.list_node()
        print(f"   ✓ اتصال به K8s API برقرار شد. تعداد نودها: {len(nodes.items)}")
        return True
    except Exception as e:
        print(f"   ✗ خطا در اتصال به K8s API: {e}")
        print("     (مطمئن شوید ~/.kube/config روی این ماشین به کلاستر دسترسی دارد)")
        return False


def test_node_labels():
    print("۳) تست لیبل edge-server-id روی نودها ...")
    from common.config import CFG
    from k8s_adapter import k8s_client
    missing = []
    for sid in CFG.server_info:
        try:
            k8s_client._get_node_name(sid)
        except RuntimeError:
            missing.append(sid)
    if missing:
        print(f"   ✗ سرورهای بدون لیبل edge-server-id: {missing}")
        print("     دستور نمونه: kubectl label node <نام‌نود> edge-server-id=<id>")
        return False
    print("   ✓ همه‌ی ۱۰ سرور لیبل edge-server-id دارند.")
    return True


def test_deployment_roundtrip():
    print("۴) تست کامل: ساخت/انتظار/فراخوانی/حذف یک Deployment آزمایشی (سرویس ۱ روی سرور ۱) ...")
    from k8s_adapter import k8s_client
    import httpx

    try:
        k8s_client.uncordon_node(1)
        k8s_client.create_deployment(service_id=1, server_id=1)
        print("   ... Deployment ساخته شد، منتظر Ready شدن (حداکثر ۶۰ ثانیه) ...")
        start = time.monotonic()
        ip = None
        while time.monotonic() - start < 60:
            if k8s_client.is_deployment_ready(1, 1):
                ip = k8s_client.get_pod_ip(1, 1)
                if ip:
                    break
            time.sleep(2)
        if not ip:
            print("   ✗ پاد در ۶۰ ثانیه Ready نشد. با kubectl get pods -n edge-rl بررسی کنید.")
            return False
        print(f"   ✓ پاد Ready شد، IP={ip}")

        print("   ... ارسال یک درخواست واقعی HTTP ...")
        port = k8s_client.worker_port(1)
        resp = httpx.post(f"http://{ip}:{port}/process", json={"request_id": 1}, timeout=15)
        resp.raise_for_status()
        print(f"   ✓ پاسخ دریافت شد: {resp.json()}")
        return True
    except Exception as e:
        print(f"   ✗ خطا: {e}")
        return False
    finally:
        print("   ... پاک‌سازی Deployment آزمایشی ...")
        k8s_client.delete_deployment(service_id=1, server_id=1)


def main():
    results = [test_redis(), test_k8s_connection(), test_node_labels()]
    if all(results):
        results.append(test_deployment_roundtrip())
    print("\n" + ("همه‌ی تست‌ها موفق بودند ✓" if all(results) else "بعضی تست‌ها شکست خوردند ✗ - قبل از ادامه دیباگ کنید"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()