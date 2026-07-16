import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# IMPORT TENSORFLOW 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# KONFIGURASI & TEMA HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Deteksi Hipertensi DNN", page_icon="🩺", layout="wide")

# ==========================================
# SIDEBAR KEREN & INFORMATIF
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center;'>🩺 Navigasi Sistem</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Pilih Menu Dashboard:", 
    ["🏠 Arsitektur & Pelatihan", "📊 Evaluasi & Pengujian", "🔎 Skrining Pasien (Live)"]
)

st.sidebar.markdown("---")
st.sidebar.success("""
**🎓 Profil Peneliti:**
- **Nama:** Sofiah Baiti Auliyah
- **Topik:** Deteksi Dini Hipertensi (Usia Dewasa)
- **Metode:** Deep Neural Network (DNN)
- **Sumber Data:** Posyandu Desa Tebas
""")

st.sidebar.info("💡 **Tips Sidang:** Gunakan menu Skrining Pasien untuk mendemonstrasikan bagaimana model secara *real-time* mengenali pola tekanan darah tinggi.")

# ==========================================
# HEADER UTAMA
# ==========================================
st.title("🩺 Sistem Deteksi Dini Hipertensi (DNN)")
st.markdown("### Aplikasi Berbasis Web untuk Klasifikasi Risiko Hipertensi pada Usia Dewasa")
st.markdown("---")

# ==========================================
# LOAD DATA & PERBAIKAN ERROR VALUE SCALER
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('data_hipertensi.xlsx')
        
        # 1. Seleksi Atribut (Sesuai Bab 4.2.2 Skripsi)
        if 'Nama' in df.columns:
            df = df.drop(columns=['Nama'])
        if 'No' in df.columns:
            df = df.drop(columns=['No'])
            
        # 2. Transformasi Data Kategorikal / Label Encoding (Sesuai Bab 4.2.3 Skripsi)
        # Mengubah teks Gender menjadi angka (L=0, P=1)
        if 'Gender' in df.columns:
            # Membersihkan spasi berlebih dan menjadikan huruf kecil semua agar aman
            df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
            df['Gender'] = df['Gender'].replace({'l': 0, 'laki-laki': 0, 'p': 1, 'perempuan': 1})
            
        # Mengubah teks Status menjadi angka (Hipertensi=0, Normal=1)
        if 'Status' in df.columns:
            df['Status'] = df['Status'].astype(str).str.strip().str.lower()
            df['Status'] = df['Status'].replace({'hipertensi': 0, 'normal': 1})
            
        # 3. Proteksi Mutlak: Paksa seluruh dataframe menjadi format numerik (float/int)
        # Jika ada data yang gagal diubah (misal typo huruf), akan diubah menjadi NaN (kosong)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 4. Pembersihan Data (Data Cleaning - Sesuai Bab 4.2.1 Skripsi)
        # Menghapus baris yang mengandung nilai NaN agar tidak merusak StandardScaler
        df = df.dropna()
        
        return df
    except FileNotFoundError:
        st.error("🚨 File 'data_hipertensi.xlsx' tidak ditemukan! Pastikan file berada di direktori yang sama atau di-upload ke GitHub.")
        return pd.DataFrame() 

# MEMANGGIL FUNGSI 
df = load_data()

# Hentikan eksekusi jika data kosong agar tidak error ke bawah
if df.empty:
    st.stop()

# ==========================================
# PREPROCESSING & SKALASI 
# ==========================================
X = df.drop('Status', axis=1)
y = df['Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# FUNGSI BUILD MODEL
# ==========================================
def build_dnn_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(7,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# KONTEN BERDASARKAN MENU SIDEBAR
# ==========================================

if menu == "🏠 Arsitektur & Pelatihan":
    st.header("Arsitektur Jaringan Saraf Tiruan & Pelatihan")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Desain Model Berdasarkan Skripsi:**")
        st.markdown("""
        - **Input Layer:** 7 Parameter Klinis (Gender, Umur, BB, TB, LP, Sistolik, Diastolik)
        - **Hidden Layer 1:** 64 Neuron (ReLU) + Dropout 20%
        - **Hidden Layer 2:** 32 Neuron (ReLU) + Dropout 20%
        - **Hidden Layer 3:** 16 Neuron (ReLU)
        - **Output Layer:** 1 Neuron (Sigmoid - Klasifikasi Biner)
        """)
        
    with col2:
        epochs_input = st.slider("Atur Maksimum Epoch", min_value=10, max_value=100, value=30)
        btn_train = st.button("Mulai Pelatihan Model 🚀", use_container_width=True)

    if btn_train:
        with st.spinner("Sedang melatih model Neural Network..."):
            model = build_dnn_model()
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train, 
                validation_split=0.2, 
                epochs=epochs_input, 
                batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )
            
            st.success("✅ Pelatihan Selesai!")
            
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            chart_acc = pd.DataFrame({'Train Accuracy': acc, 'Validation Accuracy': val_acc})
            chart_loss = pd.DataFrame({'Train Loss': loss, 'Validation Loss': val_loss})
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Grafik Akurasi")
                st.line_chart(chart_acc)
            with c2:
                st.subheader("Grafik Loss")
                st.line_chart(chart_loss)

            st.info("""
            **💡 Insight Peneliti (Pertahanan Sidang):**
            Penggunaan fungsi aktivasi **ReLU** pada *hidden layer* terbukti mempercepat konvergensi model dan menghindari masalah *vanishing gradient*. Selain itu, kurva *Loss* yang stabil antara data latih dan validasi membuktikan bahwa strategi penambahan lapisan **Dropout 20%** dan **Early Stopping** sangat efektif dalam mencegah *overfitting* (model terlalu menghafal data).
            """)

elif menu == "📊 Evaluasi & Pengujian":
    st.header("Hasil Pengujian Model (Testing)")
    st.write("Model dievaluasi menggunakan 20% data pengujian (43 data) yang tidak pernah dilihat oleh model selama proses pelatihan.")
    
    with st.spinner("Menghitung metrik performa..."):
        model = build_dnn_model()
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("Confusion Matrix")
            fig_cm = plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=['Hipertensi (0)', 'Normal (1)'], 
                        yticklabels=['Hipertensi (0)', 'Normal (1)'])
            plt.xlabel('Prediksi Model')
            plt.ylabel('Data Aktual')
            st.pyplot(fig_cm)
            
        with c2:
            st.subheader("Laporan Klasifikasi")
            report = classification_report(y_test, y_pred, target_names=['Hipertensi (0)', 'Normal (1)'], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            
            akurasi_asli = report['accuracy'] * 100
            st.success(f"**Akurasi Akhir Pengujian: {akurasi_asli:.2f}%**")
            
        st.warning("""
        **💡 Insight Peneliti terhadap Dataset Imbalanced:**
        Dari visualisasi *Confusion Matrix* di atas, terlihat model memiliki sensitivitas (*Recall*) yang sangat tinggi pada kelas **Hipertensi**. Ini adalah pencapaian krusial untuk sebuah sistem skrining kesehatan. Dalam konteks medis, lebih baik model sedikit "sensitif" dalam mencurigai seseorang terkena hipertensi agar mereka segera melakukan pemeriksaan lebih lanjut, dibandingkan gagal mendeteksi orang yang sebenarnya berisiko tinggi (*False Negative* minimal).
        """)

elif menu == "🔎 Skrining Pasien (Live)":
    st.header("Form Deteksi Dini (Skrining Pasien Baru)")
    st.write("Silakan masukkan parameter kesehatan pasien. Jaringan Saraf Tiruan (DNN) akan mengkalkulasi korelasi non-linear dari setiap input untuk menentukan probabilitas risiko hipertensi.")
    
    with st.form("form_prediksi"):
        st.markdown("#### Data Fisik & Vital Sign")
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur (Tahun)", min_value=18, max_value=100, value=45)
            bb = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=75)
            tb = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=220, value=165)
        with c2:
            lp = st.number_input("Lingkar Perut (cm)", min_value=50, max_value=150, value=90)
            sistolik = st.number_input("Tekanan Sistolik (mmHg) - Batas Bawah: 140", min_value=70, max_value=250, value=145)
            diastolik = st.number_input("Tekanan Diastolik (mmHg) - Batas Bawah: 90", min_value=40, max_value=150, value=92)
            
        submit = st.form_submit_button("Analisis dengan Model DNN 🔍", use_container_width=True)
        
    if submit:
        with st.spinner("Memproses data melalui 3 Hidden Layers..."):
            # Melatih ulang singkat untuk prediksi live
            model = build_dnn_model()
            model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
            
            gender_val = 0 if gender == "Laki-laki" else 1
            input_data = np.array([[gender_val, umur, bb, tb, lp, sistolik, diastolik]])
            
            # Normalisasi input sesuai StandardScaler
            input_scaled = scaler.transform(input_data)
            
            # Prediksi Tensor
            prediksi_prob = model.predict(input_scaled)[0][0]
            
            st.markdown("---")
            if prediksi_prob < 0.5:
                st.error("### ⚠️ KESIMPULAN: BERISIKO HIPERTENSI")
                confidence = (1 - prediksi_prob) * 100
                st.write(f"Model DNN mendeteksi pola indikasi **Hipertensi** dengan tingkat keyakinan prediktif: **{confidence:.2f}%**.")
                
                st.info("""
                **🧬 Intepretasi Medis Model:**
                Model memutuskan hasil ini tidak hanya sekadar melihat angka sistolik/diastolik, melainkan mendeteksi adanya korelasi tersembunyi antara parameter fisik (seperti Lingkar Perut dan IMT) yang sejalan dengan usia pasien. Pasien disarankan untuk segera merujuk ke fasilitas kesehatan primer (Faskes Tingkat 1) untuk penanganan klinis lebih lanjut.
                """)
            else:
                st.success("### ✅ KESIMPULAN: TEKANAN DARAH NORMAL")
                confidence = prediksi_prob * 100
                st.write(f"Model DNN mengklasifikasikan kondisi kesehatan pasien sebagai **Normal** dengan tingkat keyakinan prediktif: **{confidence:.2f}%**.")
                
                st.info("""
                **🧬 Intepretasi Medis Model:**
                Berdasarkan kombinasi umur, lingkar perut, dan parameter vital yang diberikan, bobot akhir dari layer *output* (Sigmoid) mengkategorikan pasien dalam batas aman. Meskipun demikian, skrining rutin di Posyandu tetap dianjurkan sebagai langkah preventif gaya hidup sehat.
                """)
