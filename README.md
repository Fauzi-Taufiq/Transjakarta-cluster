# Analisis Pola Kepadatan Penumpang Transjakarta

Aplikasi web untuk menganalisis dan memprediksi kepadatan penumpang Transjakarta menggunakan algoritma K-Means Clustering.

## Fitur

- **Prediksi Kepadatan**: Menganalisis kepadatan penumpang berdasarkan koridor asal, tujuan, dan waktu
- **Klasifikasi Cluster**: Mengelompokkan tingkat kepadatan menjadi 4 kategori (Rendah, Sedang, Tinggi, Sangat Tinggi)
- **Rekomendasi**: Memberikan saran berdasarkan hasil analisis
- **Statistik Detail**: Menampilkan informasi lengkap tentang transaksi, pendapatan, dan durasi perjalanan
- **Interface Modern**: UI yang responsif dan user-friendly

## Instalasi

1. Clone repository ini
2. Install dependensi:
```bash
pip install -r requirements.txt
```

## Penggunaan

1. Jalankan aplikasi:
```bash
python app.py
```

2. Buka browser dan akses `http://localhost:5000`

3. Pilih:
   - Koridor Asal
   - Koridor Tujuan  
   - Waktu Perjalanan

4. Klik "Analisis Kepadatan" untuk melihat hasil prediksi

## Troubleshooting

### Error Model Loading
Jika Anda melihat error seperti "invalid load key" atau "STACK_GLOBAL requires str" di terminal, ini berarti file model pickle rusak atau tidak kompatibel. Aplikasi akan tetap berjalan dalam mode "Data-Only".

Untuk memperbaiki model, jalankan:
```bash
python fix_model.py
```

Script ini akan:
- Membuat backup file model yang rusak
- Membuat model K-Means baru dari data yang ada
- Menyimpan model dan scaler yang baru

Setelah menjalankan script perbaikan, restart aplikasi untuk menggunakan model yang baru.

## Struktur Data

Aplikasi menggunakan 3 file utama:
- `transjakarta_density_model.pkl`: Model K-Means yang sudah dilatih
- `scaler_transjakarta.pkl`: Scaler untuk normalisasi data
- `transjakarta_clustering_results.csv`: Data hasil clustering

## Kategori Kepadatan

- **Cluster 0**: Kepadatan Rendah - Layanan normal, tidak ada kemacetan signifikan
- **Cluster 1**: Kepadatan Sedang - Beberapa penumpang, layanan tetap lancar  
- **Cluster 2**: Kepadatan Tinggi - Banyak penumpang, kemungkinan ada antrian
- **Cluster 3**: Kepadatan Sangat Tinggi - Kemacetan parah, layanan terhambat

## Teknologi

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Machine Learning**: Scikit-learn (K-Means Clustering)
- **Data Processing**: Pandas, NumPy

## API Endpoints

- `GET /`: Halaman utama aplikasi
- `POST /predict`: Endpoint untuk prediksi kepadatan
- `GET /statistics`: Endpoint untuk statistik data

## Kontribusi

Silakan buat pull request untuk kontribusi atau laporkan bug melalui issues. 