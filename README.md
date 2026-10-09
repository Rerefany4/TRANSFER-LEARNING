# P2 — Transfer Learning untuk Klasifikasi Blok Berwarna

## 1. Tujuan

Membangun model klasifikasi citra untuk mengenali tiga kelas objek, yaitu black box, green box, dan red box, menggunakan ResNet18.

## 2. Dataset

Dataset terdiri dari 156 gambar:

* Black box: 52 gambar
* Green box: 51 gambar
* Red box: 53 gambar

Dataset dibagi menjadi data training dan validation.

## 3. Metode

Tiga pendekatan dibandingkan:

1. Feature Extraction menggunakan bobot pretrained ImageNet dan melatih classifier.
2. Partial Fine-Tuning menggunakan bobot pretrained ImageNet dan melatih layer4 serta classifier.
3. Training from Scratch menggunakan bobot awal acak dan melatih seluruh model.

Setiap pendekatan dilatih selama 10 epoch.

## 4. Hasil Eksperimen

| Metode                | Best Validation Accuracy |  Training Time |
| --------------------- | -----------------------: | -------------: |
| Feature Extraction    |                   84.85% |  63.32 seconds |
| Partial Fine-Tuning   |                   90.91% |  74.07 seconds |
| Training from Scratch |                   81.82% | 122.57 seconds |

## 5. Pengukuran Latency

Pengukuran dilakukan pada CPU menggunakan model Partial Fine-Tuning.

* Mean latency: 28.34 ms
* Median latency: 28.37 ms
* Approximate throughput: 35.29 images/second

Pengukuran mencakup forward pass model pada tensor gambar yang sudah diproses, bukan seluruh pipeline kamera.

## 6. Analisis

Partial Fine-Tuning memperoleh akurasi validation tertinggi sebesar 90.91%, diikuti Feature Extraction sebesar 84.85% dan Training from Scratch sebesar 81.82%.

Training from Scratch membutuhkan waktu paling lama. Hasil ini menunjukkan bahwa penggunaan bobot pretrained memberikan keuntungan pada eksperimen dengan dataset yang relatif kecil.

Namun, validation hanya terdiri dari 33 gambar. Oleh karena itu, hasil akurasi masih perlu dikonfirmasi menggunakan data uji terpisah yang lebih banyak dan bervariasi.

## 7. File Hasil

Model tersimpan di folder `results/`. Grafik akurasi dan ringkasan hasil setiap metode juga tersedia di folder tersebut.
