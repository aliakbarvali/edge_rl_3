"""
build_push_pull_worker.py

Build + push ایمیج edge-worker (k8s_adapter/worker_service/) به رجیستری خصوصی
محلی (192.168.1.30:5000, plain-http) و pull آن روی هر ۱۰ worker node (ctr).

برخلاف اسکریپت پروژه‌ی قبلی، اینجا هیچ YAML استاتیکی apply نمی‌شود چون
k8s_adapter/k8s_client.py خودش هر Deployment را در زمان اجرا (به‌صورت
دینامیک، به ازای هر service_id/server_id) با Kubernetes Python client
می‌سازد؛ کافیست ایمیج روی همه‌ی نودها موجود باشد.

پیش‌نیاز: containerd روی هر ۱۱ ماشین (master+10 worker) باید قبلاً برای
پذیرش رجیستری insecure/plain-http در 192.168.1.30:5000 پیکربندی شده باشد
(طبق پروژه‌ی قبلی‌تان که همین رجیستری را استفاده می‌کردید).

اجرا:
    python3 build_push_pull_worker.py
"""

import subprocess
import paramiko
import os

# ================= Configuration =================
# *** این اسکریپت روی ماشین base (192.168.1.30) اجرا می‌شود - همان‌جا که
# Docker/رجیستری محلی و Redis از قبل مستقرند. مسیر را به‌جایی که پروژه‌ی
# edge_rl را روی همین ماشین clone/کپی کرده‌اید عوض کنید.
base_dir = "/home/ali/edge_rl_3/k8s_adapter/worker_service"
docker_image = "192.168.1.30:5000/edge-worker:latest"

# *** فقط ۱۰ worker node (192.168.1.11 تا .20) - master (192.168.1.10) اینجا
# نیست چون هیچ پاد سرویسی رویش زمان‌بندی نمی‌شود (طبق تأیید شما: پادها فقط
# روی workerها). اگر بعداً به هر دلیل خواستید پادی روی master هم اجرا شود
# (که با معماری فعلی پروژه سازگار نیست، چون NODE_LABEL_KEY فقط روی
# worker1..10 ست شده)، همین‌جا 192.168.1.10 را اضافه کنید.
worker_nodes = [
    "192.168.1.11",
    "192.168.1.12",
    "192.168.1.13",
    "192.168.1.14",
    "192.168.1.15",
    "192.168.1.16",
    "192.168.1.17",
    "192.168.1.18",
    "192.168.1.19",
    "192.168.1.20",
]

ssh_user = "ali"
ssh_key_path = "/home/ali/.ssh/id_rsa_edge"
SUDO_PASS = "123"  # *** مطابق اسکریپت قبلی؛ در صورت امکان بعداً به NOPASSWD/sudoers تغییر دهید


def run_local(cmd: str):
    print(f"$ {cmd}")
    result = subprocess.run(f"echo '{SUDO_PASS}' | sudo -S {cmd}", shell=True, cwd=base_dir)
    if result.returncode != 0:
        raise SystemExit(f"دستور local شکست خورد (کد {result.returncode}): {cmd}")


def run_ssh_command(host: str, command: str):
    """
    *** رفع باگ مهم: نسخه‌ی قبلی موفقیت/شکست را از روی خالی‌بودن stderr
    تشخیص می‌داد؛ ولی sudo -S همیشه پیام «[sudo] password for ali:» را
    روی stderr می‌نویسد، حتی وقتی احراز هویت و اجرای دستور کاملاً موفق
    بوده - یعنی نسخه‌ی قبلی همیشه False Negative گزارش می‌داد. حالا از
    exit status واقعی کانال SSH (stdout.channel.recv_exit_status()) برای
    تشخیص موفقیت استفاده می‌شود؛ stderr فقط برای نمایش لاگ نگه داشته می‌شود.
    خروجی: (success: bool, out: str, err: str)
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=ssh_user, key_filename=ssh_key_path, timeout=15)
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()  # قبل از خواندن استریم‌ها، تا کامل بلاک نشویم
    out = stdout.read().decode()
    err = stderr.read().decode()
    ssh.close()
    return exit_status == 0, out, err


def main():
    # ---------------------------------------------------------------
    # 1) حذف نسخه‌ی قدیمی محلی (اگر موجود بود)
    # ---------------------------------------------------------------
    print("بررسی وجود ایمیج قبلی به‌صورت محلی ...")
    existing = subprocess.run(
        f"echo '{SUDO_PASS}' | sudo -S docker images -q {docker_image}",
        shell=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if existing:
        print(f"ایمیج {docker_image} از قبل موجود است؛ حذف می‌شود ...")
        run_local(f"docker rmi -f {docker_image}")

    # ---------------------------------------------------------------
    # 2) Build
    # ---------------------------------------------------------------
    print("در حال build ایمیج edge-worker ...")
    run_local(f"docker build --network host -t {docker_image} .")

    # ---------------------------------------------------------------
    # 3) Push به رجیستری محلی
    # ---------------------------------------------------------------
    print("در حال push به رجیستری محلی (192.168.1.30:5000) ...")
    run_local(f"docker push {docker_image}")

    # ---------------------------------------------------------------
    # 4) Pull روی همه‌ی نودها (master + 10 worker) از طریق containerd/ctr
    # ---------------------------------------------------------------
    failed_nodes = []
    for node in worker_nodes:
        print(f"--- نود {node} ---")
        print("  حذف نسخه‌ی قدیمی (اگر بود) ...")
        # حذف ممکن است چون قبلاً ایمیجی نبوده با کد غیرصفر برگردد؛ این طبیعی و بی‌ضرر است
        _ok, out, err = run_ssh_command(
            node, f"echo '{SUDO_PASS}' | sudo -S ctr -n k8s.io images remove {docker_image}"
        )

        print("  pull از رجیستری محلی ...")
        ok, out, err = run_ssh_command(
            node, f"echo '{SUDO_PASS}' | sudo -S ctr -n k8s.io images pull --plain-http {docker_image}"
        )
        if ok:
            print(f"  ✓ pull موفق روی {node}")
        else:
            print(f"  ✗ خطای واقعی در pull روی {node} (exit != 0)")
            print(f"    stdout: {out.strip()}")
            print(f"    stderr: {err.strip()}")
            failed_nodes.append(node)

    print()
    if failed_nodes:
        print(f"هشدار: pull روی {len(failed_nodes)} نود واقعاً شکست خورد: {failed_nodes}")
    else:
        print("همه‌ی نودها با موفقیت pull شدند ✓")

    print("\nحالا در k8s_adapter/k8s_client.py مقدار زیر را تنظیم کنید:")
    print(f'    WORKER_IMAGE = "{docker_image}"')


if __name__ == "__main__":
    main()