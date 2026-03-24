import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
from wordcloud import WordCloud
from collections import Counter

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, f1_score, precision_score, 
                             recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Pre-processing Setup
# ==========================================

print("⏳ Memuat pustaka Sastrawi...")
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Sumber Terpercaya: nasalsabila/kamus-alay & Riset NLP Indonesia
kamus_normalisasi = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "gk": "tidak", "g": "tidak",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "bg": "abang", "ngab": "abang", "bang": "abang", "bng": "abang",
    "gw": "saya", "gua": "saya", "lu": "kamu", "lo": "kamu",
    "kikir": "pelit", "ampas": "buruk", "cupu": "lemah", "kocak": "lucu",
    "bocil": "anak kecil", "mabar": "main bareng", "prik": "aneh",
    "aja": "saja", "aj": "saja", "pake": "pakai", "kalo": "kalau", "klo": "kalau", "elek": "jelek"
}

def normalisasi(text):
    words = text.split()
    # Mengonversi kata berdasarkan kamus, jika tidak ada tetap gunakan kata asli
    normalized_words = [str(kamus_normalisasi.get(word, word)) for word in words]
    return " ".join(normalized_words)

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text) 
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = normalisasi(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    return text

# ==========================================
# 2. Load Dataset & Train-Test Split (80:20)
# ==========================================

print("📂 Memuat dataset...")
try:
    df_full = pd.read_csv('vod_raw.csv')
    df_full['label'] = pd.to_numeric(df_full['label'], errors='coerce').fillna(0).astype(int)
    print(f"   ✅ Total data dimuat: {len(df_full)} baris")
    
    # Split data agar evaluasi valid sesuai standar riset
    df_train, df_test = train_test_split(
        df_full, test_size=0.20, random_state=42, stratify=df_full['label']
    )
    print(f"   ✅ Data dibagi: {len(df_train)} Training | {len(df_test)} Testing")
except FileNotFoundError:
    print("   ❌ Error: File 'vod_raw.csv' tidak ditemukan.")
    exit()

# ==========================================
# 3. Run Preprocessing
# ==========================================

print("⏳ Sedang melakukan text preprocessing pada kedua set data...")
df_train = df_train.copy()
df_test = df_test.copy()
df_train['clean_text'] = df_train['text'].apply(preprocess_text)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

# ==========================================
# 4. Pembobotan TF-IDF
# ==========================================

print("🧮 Menghitung vektor TF-IDF (Unigram & Bigram)...")
start_tfidf = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(df_train['clean_text']) 
y_train = df_train['label']
X_test = vectorizer.transform(df_test['clean_text'])
y_test = df_test['label']
duration_tfidf = time.time() - start_tfidf

# ==========================================
# 5. Pelatihan & Uji Model
# ==========================================

print("\n🌲 Melatih Random Forest...")
start_rf = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
# thresholding manual (jika ingin menggunakan threshold khusus, misal 0.25)
# rf_probs = rf_model.predict_proba(X_test)[:, 1]
# rf_pred = (rf_probs >= 0.25).astype(int)

duration_rf = time.time() - start_rf
df_test['RF_Prediction'] = rf_pred

print("🚀 Melatih XGBoost...")
start_xgb = time.time()
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
# thresholding manual (jika ingin menggunakan threshold khusus, misal 0.25)
# xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
# xgb_pred = (xgb_probs >= 0.25).astype(int)

duration_xgb = time.time() - start_xgb
df_test['XGB_Prediction'] = xgb_pred

# LOG WAKTU
print("\n" + "="*40)
print("⏱️  DURASI WAKTU PROSES KOMPUTASI")
print("="*40)
print(f"TF-IDF Vectorization : {duration_tfidf:.4f} detik")
print(f"Random Forest        : {duration_rf:.4f} detik")
print(f"XGBoost              : {duration_xgb:.4f} detik")
print("="*40)

# ==========================================
# 6. Evaluasi & Laporan Hasil (Format Persis Contoh)
# ==========================================

print("\n" + "="*40)
print("📊 HASIL EVALUASI (GROUND TRUTH TERSEDIA)")
print("="*40)

print("\n--- [1] Random Forest Report ---")
print(classification_report(y_test, rf_pred, target_names=['Netral', 'Hate Speech']))

print("\n--- [2] XGBoost Report ---")
print(classification_report(y_test, xgb_pred, target_names=['Netral', 'Hate Speech']))

# EKSTRAKSI METRIK KHUSUS KELAS HATE SPEECH (1)
rf_prec = precision_score(y_test, rf_pred, pos_label=1, zero_division=0)
rf_rec = recall_score(y_test, rf_pred, pos_label=1, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, pos_label=1, zero_division=0)

xgb_prec = precision_score(y_test, xgb_pred, pos_label=1, zero_division=0)
xgb_rec = recall_score(y_test, xgb_pred, pos_label=1, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, pos_label=1, zero_division=0)

# ==========================================
# 7. Visualisasi (Format Persis Contoh)
# ==========================================

print("\n📈 Membuat Grafik Perbandingan Metrik...")
labels = ['Precision', 'Recall', 'F1-Score']
rf_scores = np.array([rf_prec, rf_rec, rf_f1])
xgb_scores = np.array([xgb_prec, xgb_rec, xgb_f1])

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#2ca02c')
rects2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='#d62728')

ax.set_ylabel('Skor (0.0 - 1.0)')
ax.set_title('Komparasi Metrik Kelas Hate Speech\nRandom Forest vs XGBoost', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')
plt.tight_layout()
plt.savefig('Grafik_Komparasi_HateSpeech.png', dpi=300)
plt.show()

print("📉 Membuat Heatmap Confusion Matrix...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Netral', 'Hate Speech'], yticklabels=['Netral', 'Hate Speech'], ax=axes[0])
axes[0].set_title('Confusion Matrix - Random Forest', fontweight='bold')
axes[0].set_xlabel('Prediksi Model')
axes[0].set_ylabel('Aktual (Ground Truth)')

sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Netral', 'Hate Speech'], yticklabels=['Netral', 'Hate Speech'], ax=axes[1])
axes[1].set_title('Confusion Matrix - XGBoost', fontweight='bold')
axes[1].set_xlabel('Prediksi Model')
axes[1].set_ylabel('Aktual (Ground Truth)')
plt.tight_layout()
plt.savefig('Grafik_Confusion_Matrix.png', dpi=300)
plt.show()

# ==========================================
# 8. Word Cloud & Top Hate Word (Training vs Testing Terpisah)
# ==========================================

print("\n☁️ Menghasilkan Word Cloud & Top Words untuk Hate Speech (Terpisah)...")

df_train_hate = df_train[df_train['label'] == 1]
df_test_hate = df_test[df_test['label'] == 1]

if not df_train_hate.empty or not df_test_hate.empty:
    # Ekstraksi teks
    train_text = " ".join(df_train_hate['clean_text'].astype(str)) if not df_train_hate.empty else ""
    test_text = " ".join(df_test_hate['clean_text'].astype(str)) if not df_test_hate.empty else ""

    # --- [A] Word Cloud (Subplots) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if train_text.strip():
        wc_train = WordCloud(width=600, height=400, background_color='white', colormap='Reds', max_words=100).generate(train_text)
        axes[0].imshow(wc_train, interpolation='bilinear')
        axes[0].set_title('Word Cloud - Data Training', fontsize=14, fontweight='bold')
    else:
        axes[0].set_title('Word Cloud - Data Training (Kosong)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    if test_text.strip():
        wc_test = WordCloud(width=600, height=400, background_color='white', colormap='Reds', max_words=100).generate(test_text)
        axes[1].imshow(wc_test, interpolation='bilinear')
        axes[1].set_title('Word Cloud - Data Uji', fontsize=14, fontweight='bold')
    else:
        axes[1].set_title('Word Cloud - Data Uji (Kosong)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('WordCloud_HateSpeech_Split.png', dpi=300)
    plt.show()

    # --- [B] Top 10 Hate Words (Subplots) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if train_text.strip():
        top_10_train = Counter(train_text.split()).most_common(10)
        df_top_train = pd.DataFrame(top_10_train, columns=['Kata', 'Frekuensi'])
        sns.barplot(x='Frekuensi', y='Kata', data=df_top_train, palette='flare', ax=axes[0])
        axes[0].set_title('Top 10 Kata - Data Training', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', linestyle='--', alpha=0.6)

    if test_text.strip():
        top_10_test = Counter(test_text.split()).most_common(10)
        df_top_test = pd.DataFrame(top_10_test, columns=['Kata', 'Frekuensi'])
        sns.barplot(x='Frekuensi', y='Kata', data=df_top_test, palette='flare', ax=axes[1])
        axes[1].set_title('Top 10 Kata - Data Uji', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('Top_Words_HateSpeech_Split.png', dpi=300)
    plt.show()
else:
    print("⚠️ Tidak ada data Hate Speech yang ditemukan untuk visualisasi.")

# Kesimpulan Akhir
print("\n🏆 KESIMPULAN AKHIR:")
if rf_f1 > xgb_f1:
    print(f"Random Forest lebih unggul {(rf_f1 - xgb_f1):.4f} poin pada F1-Score dibanding XGBoost.")
elif xgb_f1 > rf_f1:
    print(f"XGBoost lebih unggul {(xgb_f1 - rf_f1):.4f} poin pada F1-Score dibanding Random Forest.")
else:
    print("Kedua model memiliki F1-Score yang setara.")

# ==========================================
# 9. Simpan Hasil
# ==========================================
output_filename = 'hasil_klasifikasi_livestream.csv'
df_test.to_csv(output_filename, index=False)
print(f"\n💾 Hasil prediksi lengkap disimpan ke: {output_filename}")
print("🖼️ File grafik berhasil disimpan: 'Grafik_Komparasi_HateSpeech.png', 'Grafik_Confusion_Matrix.png', 'WordCloud_HateSpeech_Split.png', & 'Top_Words_HateSpeech_Split.png'")