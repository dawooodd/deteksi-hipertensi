# 🩺 Sistem Deteksi Dini Hipertensi Menggunakan Deep Neural Network (DNN)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[link-streamlit-anda].streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository ini berisi implementasi kode untuk skripsi dengan judul **"Deteksi Dini Hipertensi Pada Usia Dewasa Menggunakan Deep Neural Network (DNN) dari Data Skrining Kesehatan"**[cite: 1]. Aplikasi ini dibangun menggunakan antarmuka antarmuka interaktif berbasis web (Streamlit) yang mampu memprediksi risiko hipertensi secara *real-time* berdasarkan input parameter klinis pasien.

---

## 📖 Latar Belakang
Hipertensi merupakan penyakit penyumbang angka mortalitas yang tinggi dan sering disebut sebagai *silent killer*[cite: 1]. Deteksi dini sangat krusial untuk mencegah komplikasi serius. Penelitian ini menerapkan teknologi kecerdasan buatan, khususnya *Deep Learning*, untuk membantu proses klasifikasi atau prediksi kondisi kesehatan seseorang berdasarkan data skrining kesehatan yang sederhana dan mudah diperoleh di fasilitas kesehatan primer seperti Posyandu[cite: 1].

## 📊 Dataset Penelitian
Data yang digunakan dalam penelitian ini merupakan hasil skrining kesehatan masyarakat usia dewasa (18 - 60 tahun)[cite: 1].
* **Sumber Data:** Posyandu Desa Tebas, Kecamatan Gondangwetan, Kabupaten Pasuruan[cite: 1].
* **Jumlah Data:** 214 baris (171 data *training*, 43 data *testing*)[cite: 1].
* **Distribusi Kelas:** 167 Hipertensi (Imbalanced) dan 47 Normal[cite: 1].
* **7 Fitur Input (Prediktor):** 
  1. Gender (Jenis Kelamin)[cite: 1]
  2. Umur[cite: 1]
  3. Berat Badan (BB)[cite: 1]
  4. Tinggi Badan (TB)[cite: 1]
  5. Lingkar Perut (LP)[cite: 1]
  6. Tekanan Darah Sistolik[cite: 1]
  7. Tekanan Darah Diastolik[cite: 1]
* **Variabel Target:** Status Hipertensi (Klasifikasi Biner: 0 = Hipertensi, 1 = Normal)[cite: 1].

## 🧠 Arsitektur Model (DNN)
Model *Deep Neural Network* dirancang menggunakan *library* TensorFlow/Keras dengan spesifikasi arsitektur berikut:
* **Input Layer:** 7 parameter[cite: 1].
* **Hidden Layer 1:** 64 Neuron (Fungsi Aktivasi: ReLU) + *Dropout* (0.2)[cite: 1].
* **Hidden Layer 2:** 32 Neuron (Fungsi Aktivasi: ReLU) + *Dropout* (0.2)[cite: 1].
* **Hidden Layer 3:** 16 Neuron (Fungsi Aktivasi: ReLU)[cite: 1].
* **Output Layer:** 1 Neuron (Fungsi Aktivasi: Sigmoid)[cite: 1].
* **Optimizer:** Adam[cite: 1].
* **Loss Function:** Binary Crossentropy[cite: 1].
* **Pencegahan Overfitting:** Early Stopping (patience=10) dan Dropout layer[cite: 1].

## 📈 Hasil Evaluasi Kinerja
Setelah dilakukan pengujian terhadap 20% data *testing* (43 data), model menghasilkan metrik performa sebagai berikut:
* **Akurasi Pengujian:** 88,37%[cite: 1].
* **Precision (Kelas Hipertensi):** 94%[cite: 1].
* **Recall (Kelas Hipertensi):** 91%[cite: 1].
* **F1-Score (Kelas Hipertensi):** 93%[cite: 1].

Model menunjukkan kemampuan yang sangat baik dan sensitivitas yang tinggi dalam mengidentifikasi individu dengan risiko hipertensi secara tepat[cite: 1].

---

## 🚀 Panduan Instalasi (Menjalankan di Komputer Lokal)

Jika Anda ingin menjalankan aplikasi *dashboard* ini di komputer (PC/Laptop) Anda sendiri, ikuti langkah-langkah berikut:

**1. Clone Repository**
```bash
git clone [https://github.com/username-anda/nama-repository.git](https://github.com/username-anda/nama-repository.git)
cd nama-repository

```

**2. Buat Virtual Environment (Opsional tapi direkomendasikan)**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# MacOS/Linux:
source venv/bin/activate

```

**3. Instalasi Dependensi (Libraries)**

```bash
pip install -r requirements.txt

```

**4. Jalankan Aplikasi Streamlit**

```bash
streamlit run app.py

```

*Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501*`

---

## 📂 Struktur Direktori

```text
📦 deteksi-hipertensi-dnn
 ┣ 📜 app.py                  # Kode utama dashboard Streamlit & Model TensorFlow
 ┣ 📜 data_hipertensi.xlsx    # Dataset asli skrining kesehatan
 ┣ 📜 requirements.txt        # Daftar library dan versi untuk deployment
 ┗ 📜 README.md               # Dokumentasi project

```

## 🌐 Live Demo (Deployment)

Sistem ini telah di-*deploy* ke *Streamlit Community Cloud* dan dapat diakses publik secara *online*.
👉 **[Klik di sini untuk mencoba Live Demo Skrining Pasien]([https://www.google.com/search?q=https://%5Bubah-dengan-link-aplikasi-anda%5D.streamlit.app](https://deteksi-hipertensi.streamlit.app/))**



---


*Program Studi Teknik Informatika, Fakultas Teknologi Informasi*

*Universitas Merdeka Pasuruan*

```

```
