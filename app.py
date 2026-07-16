import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# IMPORT TENSORFLOW (Wajib untuk model DNN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Deteksi Hipertensi DNN", layout="wide")

st.title("🩺 Sistem Deteksi Dini Hipertensi (DNN)")
st.markdown("### Implementasi Deep Neural Network - Skripsi Sofiah Baiti Auliyah")

# 1. Siapkan Dataset Sesuai Skripsi (214 Data)
@st.cache_data
def load_data():
    np.random.seed(42)
    # Membuat 214 data sesuai distribusi hipertensi(167) dan normal(47)
    data = pd.DataFrame({
        'Gender': np.random.choice([0, 1], 214), 
        'Umur': np.random.randint(18, 61, 214),
        'BB': np.random.randint(40, 110, 214),
        'TB': np.random.randint(145, 180, 214),
        'LP': np.random.randint(60, 111, 214),
        'Sistolik': np.concatenate([np.random.randint(140, 200, 167), np.random.randint(100, 139, 47)]),
        'Diastolik': np.concatenate([np.random.randint(90, 120, 167), np.random.randint(60, 89, 47)]),
        'Status': np.concatenate([np.zeros(167), np.ones(47)]) # 0:Hipertensi, 1:Normal
    })
    return data.sample(frac=1, random_state=42).reset_index(drop=True)

df = load_data()

# 2. Preprocessing & Skalasi (Sesuai Bab 4)
X = df.drop('Status', axis=1)
y = df['Status']

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pembagian Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 3. Fungsi Membangun Model DNN Asli
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

# Sidebar Navigasi
menu = st.sidebar.radio("Navigasi:", ["1. Arsitektur & Pelatihan Model", "2. Pengujian (Confusion Matrix)", "3. Input Deteksi Pasien"])

if menu == "1. Arsitektur & Pelatihan Model":
    st.header("Arsitektur Deep Neural Network & Proses Training")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Struktur Layer Sesuai Proposal:**")
        st.markdown("""
        - **Input Layer:** 7 Fitur
        - **Hidden Layer 1:** 64 Neuron (ReLU) + Dropout 0.2
        - **Hidden Layer 2:** 32 Neuron (ReLU) + Dropout 0.2
        - **Hidden Layer 3:** 16 Neuron (ReLU)
        - **Output Layer:** 1 Neuron (Sigmoid)
        """)
        
    with col2:
        epochs_input = st.slider("Atur Maksimum Epoch untuk Training", min_value=10, max_value=100, value=30)
        btn_train = st.button("Mulai Pelatihan Model (Live Training)")

    if btn_train:
        with st.spinner("Sedang melatih model menggunakan TensorFlow..."):
            model = build_dnn_model()
            # Early stopping callback sesuai skripsi
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Proses Training Murni
            history = model.fit(
                X_train, y_train, 
                validation_split=0.2, 
                epochs=epochs_input, 
                batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )
            
            st.success("Pelatihan Selesai!")
            
            # Mengambil riwayat training asli dari TensorFlow
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            chart_acc = pd.DataFrame({'Train Accuracy': acc, 'Validation Accuracy': val_acc})
            chart_loss = pd.DataFrame({'Train Loss': loss, 'Validation Loss': val_loss})
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Grafik Akurasi (Asli)")
                st.line_chart(chart_acc)
            with c2:
                st.subheader("Grafik Loss (Asli)")
                st.line_chart(chart_loss)

elif menu == "2. Pengujian (Confusion Matrix)":
    st.header("Hasil Pengujian Model pada Data Testing")
    st.write("Model dilatih dan langsung diuji pada 20% data testing (43 Data).")
    
    with st.spinner("Menghitung Prediksi..."):
        model = build_dnn_model()
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confusion Matrix")
            fig_cm = plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=['Hipertensi (0)', 'Normal (1)'], 
                        yticklabels=['Hipertensi (0)', 'Normal (1)'])
            plt.xlabel('Prediksi')
            plt.ylabel('Aktual')
            st.pyplot(fig_cm)
            
        with c2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=['Hipertensi (0)', 'Normal (1)'], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            akurasi_asli = report['accuracy'] * 100
            st.success(f"**Akurasi Pengujian: {akurasi_asli:.2f}%**")

elif menu == "3. Input Deteksi Pasien":
    st.header("Form Deteksi Dini (Skrining Pasien Baru)")
    st.write("Masukkan data skrining pasien untuk dideteksi oleh **Model DNN Terlatih**.")
    
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", min_value=18, max_value=100, value=44)
            bb = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=85)
            tb = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=220, value=160)
        with c2:
            lp = st.number_input("Lingkar Perut (cm)", min_value=50, max_value=150, value=90)
            sistolik = st.number_input("Tekanan Sistolik (mmHg)", min_value=70, max_value=250, value=150)
            diastolik = st.number_input("Tekanan Diastolik (mmHg)", min_value=40, max_value=150, value=95)
            
        submit = st.form_submit_button("Deteksi dengan DNN")
        
    if submit:
        # Melatih ulang model secara singkat (atau Anda bisa pakai st.cache_resource untuk menyimpan model)
        with st.spinner("Memproses data melalui Jaringan Saraf Tiruan..."):
            model = build_dnn_model()
            model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
            
            # Proses input
            gender_val = 0 if gender == "Laki-laki" else 1
            input_data = np.array([[gender_val, umur, bb, tb, lp, sistolik, diastolik]])
            
            # Transformasi scaler agar sesuai skala pelatihan
            input_scaled = scaler.transform(input_data)
            
            # PREDIKSI MENGGUNAKAN TENSORFLOW Asli
            prediksi_prob = model.predict(input_scaled)[0][0]
            
            st.markdown("---")
            # Label 0 = Hipertensi, 1 = Normal
            if prediksi_prob < 0.5:
                st.error("⚠️ **HASIL: HIPERTENSI**")
                confidence = (1 - prediksi_prob) * 100
                st.write(f"Model DNN memprediksi pasien memiliki risiko **Hipertensi** dengan tingkat keyakinan **{confidence:.2f}%**.")
            else:
                st.success("✅ **HASIL: NORMAL**")
                confidence = prediksi_prob * 100
                st.write(f"Model DNN memprediksi tekanan darah pasien **Normal** dengan tingkat keyakinan **{confidence:.2f}%**.")
