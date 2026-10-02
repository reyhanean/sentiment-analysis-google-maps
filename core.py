"""
core.py — "OTAK" aplikasi.

Semua logika machine learning ada di sini: preprocessing, training,
evaluasi, dan prediksi. Tidak ada ketergantungan ke Colab / model manapun
dari luar — semuanya dilatih dari dataset mentah yang diupload lewat web,
di dalam aplikasi ini sendiri.

Pipeline (replikasi dari notebook penelitian):
  1. Load dataset, hapus baris kosong & duplikat
  2. Case Folding
  3. Cleansing (URL, mention, hashtag, angka, emoji, tanda baca)
  4. Normalisasi Slang
  5. Tokenizing (NLTK)
  6. Stopword Removal (Sastrawi)
  7. Stemming (Sastrawi)
  8. Labeling (bintang -> Positif/Netral/Negatif)
  9. Split 80/20 stratified
  10. Manual Random Oversampling (data training saja)
  11. TF-IDF (ngram 1-2, max_features=5000)
  12. Training ComplementNB (alpha=0.3)
  13. Evaluasi + simpan grafik (distribusi, confusion matrix, top kata, word cloud)
"""

import os
import re
import json
import string
from collections import Counter

import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# ---------------------------------------------------------------
# Path penyimpanan
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
CHARTS_DIR = os.path.join(BASE_DIR, 'static', 'charts')
MODEL_PATH = os.path.join(STORAGE_DIR, 'model_cnb.pkl')
TFIDF_PATH = os.path.join(STORAGE_DIR, 'tfidf.pkl')
DATASET_PATH = os.path.join(STORAGE_DIR, 'dataset_final.csv')
METRICS_PATH = os.path.join(STORAGE_DIR, 'metrics.json')

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Siapkan NLTK (sekali saja, otomatis download kalau belum ada)
# ---------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------
# Kamus slang & alat Sastrawi (dipakai berulang)
# ---------------------------------------------------------------
KAMUS_SLANG = {
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'tdk': 'tidak', 'bgt': 'banget', 'bgtt': 'banget', 'yg': 'yang', 'dg': 'dengan',
    'dgn': 'dengan', 'utk': 'untuk', 'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
    'blm': 'belum', 'bl': 'belum', 'pd': 'pada', 'tp': 'tapi', 'tetep': 'tetap',
    'jgn': 'jangan', 'jd': 'jadi', 'jadinya': 'jadi', 'krn': 'karena', 'karna': 'karena',
    'kl': 'kalau', 'klo': 'kalau', 'kalo': 'kalau', 'sy': 'saya', 'gw': 'saya',
    'gue': 'saya', 'lu': 'kamu', 'lo': 'kamu', 'org': 'orang', 'byk': 'banyak',
    'bnyk': 'banyak', 'dr': 'dari', 'skrg': 'sekarang', 'skg': 'sekarang', 'mksh': 'terima kasih',
    'makasih': 'terima kasih', 'trs': 'terus', 'terusan': 'terus', 'sm': 'sama',
    'jg': 'juga', 'jgk': 'juga', 'emg': 'memang', 'emang': 'memang', 'gt': 'begitu',
    'gitu': 'begitu', 'sbnrnya': 'sebenarnya', 'sbnrnya2': 'sebenarnya', 'wkwk': 'haha',
    'wkwkwk': 'haha', 'parah': 'sangat', 'bener': 'benar', 'bnr': 'benar', 'sesuatu2': 'sesuatu'
}

_stopword_list = set(StopWordRemoverFactory().get_stop_words())
_stemmer = StemmerFactory().create_stemmer()
_stem_cache = {}


def _stem_cached(word: str) -> str:
    """Stem satu kata, dengan cache supaya kata yang sama tidak diproses
    ulang dari nol (Sastrawi cukup lambat per kata, dan ulasan pelanggan
    banyak memakai kata yang berulang seperti 'pelayanan', 'ramah', dst)."""
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]

SENTIMENT_COLORS = {'Positif': '#4C7A51', 'Netral': '#C9A227', 'Negatif': '#8B1E3F'}


# ---------------------------------------------------------------
# 6 Tahap Preprocessing
# ---------------------------------------------------------------
def case_folding(text) -> str:
    return str(text).lower()


def cleansing(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.encode('ascii', 'ignore').decode()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalisasi(text: str) -> str:
    words = text.split()
    return ' '.join(KAMUS_SLANG.get(w, w) for w in words)


def tokenizing(text: str) -> list:
    return word_tokenize(text)


def stopword_removal(tokens: list) -> list:
    return [t for t in tokens if t not in _stopword_list]


def stemming(tokens: list) -> list:
    return [_stem_cached(t) for t in tokens]


def preprocess_text(raw_text: str) -> str:
    """Jalankan 6 tahap preprocessing pada satu teks, kembalikan text_clean."""
    text = case_folding(raw_text)
    text = cleansing(text)
    text = normalisasi(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return ' '.join(tokens)


def labeling(stars) -> str:
    if stars <= 2:
        return 'Negatif'
    elif stars == 3:
        return 'Netral'
    else:
        return 'Positif'


# ---------------------------------------------------------------
# Training penuh dari dataset mentah
# ---------------------------------------------------------------
def find_column(columns, candidates):
    cols_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _fig_to_file(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=140)
    plt.close(fig)
    return f'charts/{filename}'


def train_pipeline(csv_file_storage, progress_cb=None):
    """
    csv_file_storage: file-like object (dari request.files) berisi CSV mentah
                       dengan kolom teks ulasan + rating bintang.
    progress_cb: optional callable(str) untuk melaporkan progress.
    Mengembalikan dict ringkasan hasil training, dan menyimpan semua
    artefak (model, vectorizer, dataset, grafik, metrics.json) ke disk.
    """
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    # ---- 1. Load & bersihkan dataset ----
    log('Membaca dataset...')
    raw_df = pd.read_csv(csv_file_storage)

    text_col = find_column(raw_df.columns, ['text', 'ulasan', 'review', 'komentar'])
    stars_col = find_column(raw_df.columns, ['stars', 'rating', 'bintang', 'star'])
    if text_col is None or stars_col is None:
        raise ValueError(
            f'Kolom tidak dikenali. Kolom yang ditemukan: {", ".join(raw_df.columns)}. '
            'Pastikan ada kolom teks ulasan (mis. "text") dan kolom rating (mis. "stars").'
        )

    df = raw_df[[text_col, stars_col]].rename(columns={text_col: 'text', stars_col: 'stars'})
    df = df.dropna(subset=['text'])
    df = df.drop_duplicates(subset=['text'])
    df = df.reset_index(drop=True)
    log(f'Jumlah data setelah bersih-bersih: {len(df)}')

    if len(df) < 30:
        raise ValueError('Dataset terlalu sedikit (minimal ±30 baris) untuk training yang layak.')

    # ---- 2-7. Preprocessing 6 tahap ----
    log('Case folding...')
    df['case_folding'] = df['text'].apply(case_folding)
    log('Cleansing...')
    df['cleansing'] = df['case_folding'].apply(cleansing)
    log('Normalisasi slang...')
    df['normalisasi'] = df['cleansing'].apply(normalisasi)
    log('Tokenizing...')
    df['tokenizing'] = df['normalisasi'].apply(tokenizing)
    log('Stopword removal...')
    df['stopword'] = df['tokenizing'].apply(stopword_removal)
    log('Stemming (paling lama, kata yang sama otomatis di-cache)...')
    total_baris = len(df)
    hasil_stemming = []
    for i, tokens in enumerate(df['stopword']):
        hasil_stemming.append(stemming(tokens))
        if (i + 1) % 200 == 0 or (i + 1) == total_baris:
            log(f'  Stemming baris {i + 1}/{total_baris}...')
    df['stemming'] = hasil_stemming
    df['text_clean'] = df['stemming'].apply(lambda toks: ' '.join(toks))

    # ---- 8. Labeling + buang kosong ----
    log('Labeling...')
    df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
    df = df.dropna(subset=['stars'])
    df['label'] = df['stars'].apply(labeling)
    df = df[df['text_clean'].str.strip() != '']
    df = df.reset_index(drop=True)

    label_counts = df['label'].value_counts()
    if label_counts.min() < 5:
        raise ValueError(
            f'Salah satu kelas sentimen datanya terlalu sedikit (minimal 5 baris per kelas). '
            f'Distribusi saat ini: {label_counts.to_dict()}'
        )

    # ---- 9. Split 80/20 stratified ----
    log('Split data training/testing...')
    X = df['text_clean']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 10. Oversampling (loop per kelas, aman di semua versi pandas) ----
    log('Oversampling data training...')
    train_df = pd.DataFrame({'text_clean': X_train, 'label': y_train})
    jumlah_maks = train_df['label'].value_counts().max()
    frames = []
    for _, group in train_df.groupby('label'):
        frames.append(group.sample(jumlah_maks, replace=True, random_state=42))
    train_df_balanced = pd.concat(frames, ignore_index=True)
    X_train_balanced = train_df_balanced['text_clean']
    y_train_balanced = train_df_balanced['label']

    # ---- 11. TF-IDF ----
    log('TF-IDF vectorization...')
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_balanced)
    X_test_tfidf = tfidf.transform(X_test)

    # ---- 12. Training ----
    log('Training model ComplementNB...')
    model = ComplementNB(alpha=0.3)
    model.fit(X_train_tfidf, y_train_balanced)

    # ---- 13. Evaluasi ----
    log('Evaluasi model...')
    y_pred = model.predict(X_test_tfidf)
    akurasi = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    # ---- Grafik: Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Confusion Matrix')
    chart_confusion = _fig_to_file(fig, 'confusion.png')

    # ---- Grafik: Distribusi Sentimen ----
    label_urut = ['Positif', 'Netral', 'Negatif']
    distribusi = df['label'].value_counts().reindex(label_urut).fillna(0)
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    colors = [SENTIMENT_COLORS[k] for k in label_urut]
    bars = ax.bar(label_urut, distribusi.values, color=colors)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(distribusi.values) * 0.01,
                str(int(h)), ha='center', fontweight='bold')
    ax.set_ylabel('Jumlah Data')
    ax.set_title('Distribusi Sentimen Ulasan')
    ax.spines[['top', 'right']].set_visible(False)
    chart_distribusi = _fig_to_file(fig, 'distribusi.png')

    # ---- Grafik: Oversampling sebelum/sesudah ----
    sebelum = train_df['label'].value_counts().reindex(label_urut)
    sesudah = train_df_balanced['label'].value_counts().reindex(label_urut)
    xpos = np.arange(len(label_urut))
    lebar = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar(xpos - lebar / 2, sebelum.values, lebar, label='Sebelum', color='#2E75B6')
    ax.bar(xpos + lebar / 2, sesudah.values, lebar, label='Sesudah', color='#ED7D31')
    ax.set_xticks(xpos)
    ax.set_xticklabels(label_urut)
    ax.set_ylabel('Jumlah Data')
    ax.set_title('Oversampling: Sebelum vs Sesudah')
    ax.legend()
    chart_oversampling = _fig_to_file(fig, 'oversampling.png')

    # ---- Grafik: Top 20 Kata ----
    semua_kata = ' '.join(df['text_clean']).split()
    top_words = Counter(semua_kata).most_common(20)
    kata, jumlah = zip(*top_words[::-1])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(kata, jumlah, color='#2196F3')
    ax.set_xlabel('Frekuensi')
    ax.set_title('Top 20 Kata Paling Sering Muncul')
    chart_top_words = _fig_to_file(fig, 'top_words.png')

    # ---- Grafik: Word Cloud per kelas ----
    chart_wordclouds = {}
    try:
        from wordcloud import WordCloud
        for label, warna in zip(label_urut, ['Greens', 'YlOrBr', 'Reds']):
            teks_gabungan = ' '.join(df[df['label'] == label]['text_clean'])
            if not teks_gabungan.strip():
                continue
            wc = WordCloud(width=600, height=400, background_color='white',
                            colormap=warna, max_words=100).generate(teks_gabungan)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud - {label}')
            chart_wordclouds[label] = _fig_to_file(fig, f'wordcloud_{label.lower()}.png')
    except ImportError:
        pass

    # ---- Simpan model, vectorizer, dataset, metrics ----
    log('Menyimpan model & hasil...')
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, TFIDF_PATH)

    export_df = pd.DataFrame({
        'text_original': df['text'],
        'text_clean': df['text_clean'],
        'label': df['label'],
    })
    export_df.to_csv(DATASET_PATH, index=False)

    metrics = {
        'total_data': len(df),
        'jumlah_train': len(X_train),
        'jumlah_test': len(X_test),
        'jumlah_train_balanced': len(X_train_balanced),
        'distribusi_label': df['label'].value_counts().to_dict(),
        'akurasi': akurasi,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'chart_confusion': chart_confusion,
        'chart_distribusi': chart_distribusi,
        'chart_oversampling': chart_oversampling,
        'chart_top_words': chart_top_words,
        'chart_wordclouds': chart_wordclouds,
        'model_spec': {
            'algoritma': 'Complement Naive Bayes (alpha=0.3)',
            'ekstraksi_fitur': 'TF-IDF, n-gram (1,2), max_features=5000',
            'balancing': 'Manual Random Oversampling (data training saja)',
            'split': '80/20 stratified, random_state=42',
        },
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)

    log('Selesai!')
    _model_cache.clear()  # paksa reload model baru saat prediksi berikutnya
    return metrics


# ---------------------------------------------------------------
# Untuk halaman: cek status & prediksi
# ---------------------------------------------------------------
def has_trained_model() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(METRICS_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


_model_cache = {}


def _get_model_and_vectorizer():
    if 'model' not in _model_cache:
        _model_cache['model'] = joblib.load(MODEL_PATH)
        _model_cache['tfidf'] = joblib.load(TFIDF_PATH)
    return _model_cache['model'], _model_cache['tfidf']


def predict_sentiment(raw_text: str) -> dict:
    if not has_trained_model():
        raise RuntimeError('Belum ada model. Upload dataset terlebih dahulu.')
    model, tfidf = _get_model_and_vectorizer()
    clean = preprocess_text(raw_text)
    if not clean.strip():
        raise ValueError('Teks tidak mengandung kata yang bisa dianalisis setelah preprocessing.')
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    proba_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(model.classes_, proba)}
    return {'label': pred, 'proba': proba_dict, 'clean_text': clean}
