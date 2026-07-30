# k8s_adapter/worker_service/Dockerfile
#
# ساخت و push (روی سیستم خودتان که به رجیستری کانتینر دسترسی دارد):
#     docker build -t <YOUR_REGISTRY>/edge-worker:latest .
#     docker push <YOUR_REGISTRY>/edge-worker:latest
#
# یک image واحد برای هر ۱۵ سرویس - تفاوت هر سرویس فقط از طریق env var
# EXEC_TIME_SEC در Deployment مشخص می‌شود (نگاه کنید به k8s_client.py).

FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# *** --workers 1 --limit-concurrency 1: هر پاد واقعاً هم‌زمان فقط ۱
# درخواست پردازش می‌کند (طبق بخش ۱.۲ سند)، بقیه در صف TCP معطل می‌مانند.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--limit-concurrency", "1"]
