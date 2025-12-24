import pandas as pd
import numpy as np
import datetime
import os
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """Memuat data mentah dari path yang ditentukan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan!")
    return pd.read_csv(file_path).copy()

def process_automation(df):
    """Fungsi utama untuk menjalankan seluruh alur preprocessing."""
    
    # 1. DATA CLEANING (Ekstraksi angka dari kolom string)
    cols_to_extract = ['Engine', 'Max Power', 'Max Torque']
    for col in cols_to_extract:
        if col in df.columns:
            # Menggunakan regex untuk mengambil angka (termasuk desimal)
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    # 2. HANDLING DUPLICATES
    df.drop_duplicates(inplace=True)

    # 3. HANDLING MISSING VALUES
    # Mengisi kolom numerik dengan Median
    cols_to_impute = ['Engine', 'Max Power', 'Max Torque', 'Length', 'Width', 
                      'Height', 'Seating Capacity', 'Fuel Tank Capacity']
    for col in cols_to_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Mengisi Drivetrain dengan Modus
    if 'Drivetrain' in df.columns:
        df['Drivetrain'] = df['Drivetrain'].fillna(df['Drivetrain'].mode()[0])

    # 4. HANDLING OUTLIERS (IQR Method)
    # Langkah ini krusial sebelum scaling agar data tidak terdistorsi
    outlier_cols = ['Price', 'Kilometer', 'Engine', 'Max Power']
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # 5. FEATURE ENGINEERING (Menghitung Usia Mobil)
    if 'Year' in df.columns:
        current_year = datetime.datetime.now().year
        df['Car_Age'] = current_year - df['Year']
        df.drop('Year', axis=1, inplace=True)

    # 6. DATA BINNING (Kategorisasi Kilometer)
    if 'Kilometer' in df.columns:
        labels = ['Low_Mileage', 'Moderate_Mileage', 'High_Mileage']
        # Mengelompokkan kilometer menjadi 3 bagian sama besar (quantiles)
        df['Kilometer_Category'] = pd.qcut(df['Kilometer'], q=3, labels=labels)

    # 7. SCALING (Penskalaan Fitur)
    # Standarisasi untuk fitur dengan variansi besar
    std_scaler = StandardScaler()
    cols_to_std = ['Kilometer', 'Engine', 'Max Power', 'Max Torque', 'Car_Age', 'Price']
    cols_to_std = [c for c in cols_to_std if c in df.columns]
    df[cols_to_std] = std_scaler.fit_transform(df[cols_to_std])

    # Normalisasi untuk dimensi fisik (0-1)
    mm_scaler = MinMaxScaler()
    cols_to_norm = ['Length', 'Width', 'Height', 'Fuel Tank Capacity', 'Seating Capacity']
    cols_to_norm = [c for c in cols_to_norm if c in df.columns]
    df[cols_to_norm] = mm_scaler.fit_transform(df[cols_to_norm])

    # 8. ENCODING (Mengubah teks kategorikal menjadi angka)
    categorical_cols = [
        'Fuel Type', 'Transmission', 'Owner', 
        'Drivetrain', 'Seller Type', 'Kilometer_Category'
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    # Menggunakan dtype=int agar hasilnya 0/1 bukan True/False
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 9. DROP UNNECESSARY COLUMNS
    # Menghapus kolom dengan kardinalitas tinggi agar model tidak berat
    cols_to_drop = ['Make', 'Model', 'Location', 'Color']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df

def save_output(df, output_path):
    """Menyimpan file hasil akhir ke CSV."""
    # Pastikan folder tujuan ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sukses! File tersimpan di: {output_path}")

if __name__ == "__main__":
    # Menentukan path sesuai struktur folder kriteria tugas
    INPUT_PATH = "namadataset_raw/car details v4.csv"
    OUTPUT_PATH = "preprocessing/car details v4_preprocessing.csv"

    try:
        print("Memulai otomatisasi Christian_Michael_Halim...")
        raw_df = load_data(INPUT_PATH)
        final_df = process_automation(raw_df)
        save_output(final_df, OUTPUT_PATH)
        print("Proses selesai.")
    except Exception as e:
        print(f"Gagal menjalankan otomatisasi: {e}")