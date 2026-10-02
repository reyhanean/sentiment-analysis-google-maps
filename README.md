# Ramayana Sentiment Lab 
## Struktur (cuma 4 file inti)

```
ramayana_app/
├── core.py          <- OTAK: preprocessing, training, evaluasi, prediksi
├── app.py           <- INTERFACE: Flask (login/register, upload, dashboard, prediksi)
├── report_pdf.py     <- Generator laporan PDF
├── requirements.txt
├── templates/        <- 4 halaman HTML (login, register, dashboard, base)
├── static/            <- CSS, JS, dan grafik hasil (otomatis terisi)
└── storage/            <- model, dataset, database login (otomatis terisi)
```

## Cara Menjalankan

```bash
cd ramayana_app
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000` di browser.

## Cara Pakai

1. **Daftar akun** lalu **login**.
2. Di halaman utama, **upload file CSV mentah** (kolom teks ulasan + rating
   bintang 1-5, nama kolom bebas — sistem otomatis mendeteksi `text`/`ulasan`/
   `review` dan `stars`/`rating`/`bintang`).
3. Klik **"Proses & Latih Model"**. Sistem akan otomatis:
   - Membersihkan data (hapus kosong & duplikat)
   - Menjalankan 6 tahap preprocessing (case folding, cleansing, normalisasi
     slang, tokenizing, stopword removal, stemming)
   - Labeling otomatis dari rating bintang (1-2=Negatif, 3=Netral, 4-5=Positif)
   - Split 80/20, oversampling, TF-IDF, training ComplementNB (alpha=0.3)
   - Ini bisa memakan waktu beberapa menit tergantung ukuran dataset
     (tahap stemming paling lama).
4. Dashboard otomatis terisi: ringkasan metrik, tabel classification report,
   grafik distribusi sentimen, confusion matrix, grafik oversampling,
   top 20 kata, dan word cloud per kelas.
5. Coba **prediksi ulasan baru** di bagian bawah halaman.
6. Klik **"Unduh Laporan PDF"** untuk mendapat laporan siap cetak, lengkap
   dengan tanggal dan kolom tanda tangan.

## Catatan

- Setiap kali upload dataset baru, model lama otomatis ditimpa dengan yang baru.
- Minimal dataset: ±30 baris, dengan minimal 5 baris per kelas sentimen
  (Positif/Netral/Negatif) setelah preprocessing.
- Kalau nama kolom di CSV kamu tidak dikenali otomatis, ganti dulu nama
  kolomnya jadi `text` dan `stars`, atau edit fungsi `find_column()` di `core.py`.
- Aturan konversi bintang→label ada di fungsi `labeling()` di `core.py`, gampang
  diubah kalau perlu.
