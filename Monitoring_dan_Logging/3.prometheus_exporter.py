import time
import random
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# --- 1. METRIKS SISTEM (Kesehatan Laptop/Server) ---
RAM_USAGE = Gauge('server_ram_usage_percent', 'Persentase penggunaan RAM server')
CPU_USAGE = Gauge('server_cpu_usage_percent', 'Persentase penggunaan CPU server')

# --- 2. METRIKS MODEL (Kinerja Prediksi Harga Mobil) ---
PRICE_PREDICTIONS = Counter('model_car_price_predictions_total', 'Total prediksi harga mobil yang dilakukan')
MODEL_ERROR_SCORE = Gauge('model_mae_score', 'Skor Mean Absolute Error real-time (Simulasi)')
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Waktu proses prediksi (detik)')

def generate_metrics():
    print("✅ Prometheus Exporter Berjalan di http://localhost:8000")
    print("Sedang merekam data sistem dan model... Tekan CTRL+C untuk berhenti.")
    
    while True:
        # Update Metriks Sistem menggunakan psutil
        RAM_USAGE.set(psutil.virtual_memory().percent)
        CPU_USAGE.set(psutil.cpu_percent())

        # Simulasi Aktivitas Model (Random trigger seolah-olah ada user yang mengakses)
        if random.random() < 0.8: 
            PRICE_PREDICTIONS.inc()
            print("Log: 1 Prediksi Harga Mobil dicatat.")
            
            # Simulasi Latency (Waktu tunggu)
            with REQUEST_LATENCY.time():
                time.sleep(random.uniform(0.1, 0.4))

        # Simulasi MAE (Seolah-olah error model naik turun secara dinamis)
        mae_sim = 1.14 + random.uniform(-0.05, 0.05)
        MODEL_ERROR_SCORE.set(mae_sim)

        time.sleep(2) # Update setiap 2 detik

if __name__ == '__main__':
    # Menjalankan server exporter pada port 8000
    start_http_server(8000)
    generate_metrics()