import requests
import json

# Pastikan portnya sama dengan port saat Anda melakukan 'serve' (5001)
url = "http://localhost:5001/invocations"

# Daftar 29 kolom sesuai fit time model Anda
columns = [
    'Kilometer', 'Engine', 'Max Power', 'Max Torque', 'Length', 'Width', 'Height', 
    'Seating Capacity', 'Fuel Tank Capacity', 'Car_Age', 'Fuel Type_CNG + CNG', 
    'Fuel Type_Diesel', 'Fuel Type_Electric', 'Fuel Type_LPG', 'Fuel Type_Petrol', 
    'Fuel Type_Petrol + CNG', 'Fuel Type_Petrol + LPG', 'Transmission_Manual', 
    'Owner_First', 'Owner_Fourth', 'Owner_Second', 'Owner_Third', 'Owner_UnRegistered Car', 
    'Drivetrain_FWD', 'Drivetrain_RWD', 'Seller Type_Corporate', 'Seller Type_Individual', 
    'Kilometer_Category_Moderate_Mileage', 'Kilometer_Category_High_Mileage'
]

# Nilai dummy (pastikan jumlahnya tepat 29 nilai)
# Contoh: Kita set mobil bensin (Petrol), transmisi Manual, tangan Pertama (First Owner), penggerak depan (FWD)
data_values = [
    15000, 1200, 85, 113, 3995, 1735, 1515, 5, 42, 4, # 10 Kolom Numerik
    0, 0, 0, 0, 1, 0, 0,                             # Fuel Type (Hanya Petrol yang 1)
    1,                                               # Transmission_Manual (1 = Ya)
    1, 0, 0, 0, 0,                                   # Owner (Hanya First yang 1)
    1, 0,                                            # Drivetrain (Hanya FWD yang 1)
    0, 1,                                            # Seller Type (Hanya Individual yang 1)
    1, 0                                             # Kilometer Category (Moderate = 1)
]

data = {
    "dataframe_split": {
        "columns": columns,
        "data": [data_values]
    }
}

print("Sedang mengirim request ke model server...")

try:
    # Mengirim request POST ke server Uvicorn (port 5001)
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print("\n✅ BERHASIL!")
        # Mengambil hasil prediksi dari JSON response
        prediction = response.json()
        print(f"Hasil Prediksi Harga: {prediction}")
    else:
        print("\n❌ GAGAL!")
        print(f"Status Code: {response.status_code}")
        print(f"Pesan Error: {response.text}")

except Exception as e:
    print("\n❌ ERROR: Pastikan terminal 'serving' (Uvicorn) Anda masih terbuka dan aktif di port 5001!")
    print(f"Pesan Error: {str(e)}")