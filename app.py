"""
app.py — INTERFACE aplikasi (Flask).

Menangani: login/register, upload dataset mentah (lalu training lewat
core.py), menampilkan dashboard hasil, prediksi ulasan baru, dan generate
laporan PDF siap cetak.
"""

import os
import sqlite3
import secrets
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_file,
)
from werkzeug.security import generate_password_hash, check_password_hash

import core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'storage', 'users.db')
SECRET_KEY_PATH = os.path.join(BASE_DIR, 'storage', 'secret_key.txt')

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ---------------------------------------------------------------
# Setup Flask
# ---------------------------------------------------------------
def get_or_create_secret_key():
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(SECRET_KEY_PATH, 'w') as f:
        f.write(key)
    return key


app = Flask(__name__)
app.secret_key = get_or_create_secret_key()
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # maks 20MB upload


# ---------------------------------------------------------------
# Database sederhana untuk login/register (SQLite)
# ---------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------
# Auth: Register, Login, Logout
# ---------------------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        if not username or not password:
            flash('Username dan password wajib diisi.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password minimal 6 karakter.', 'error')
            return render_template('register.html')
        if password != confirm:
            flash('Konfirmasi password tidak cocok.', 'error')
            return render_template('register.html')

        conn = get_db()
        existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if existing:
            conn.close()
            flash('Username sudah dipakai, coba yang lain.', 'error')
            return render_template('register.html')

        conn.execute(
            'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
            (username, generate_password_hash(password), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        flash('Akun berhasil dibuat. Silakan login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user is None or not check_password_hash(user['password_hash'], password):
            flash('Username atau password salah.', 'error')
            return render_template('login.html')

        session['user_id'] = user['id']
        session['username'] = user['username']
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------------------------------------------------------
# Halaman utama: upload + dashboard hasil + prediksi
# ---------------------------------------------------------------
@app.route('/')
@login_required
def predict_page():
    trained = core.has_trained_model()
    return render_template('predict.html', trained=trained, username=session.get('username'))


@app.route('/dashboard')
@login_required
def dashboard():
    metrics = core.load_metrics()
    return render_template('dashboard.html', metrics=metrics, username=session.get('username'))


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files.get('dataset_file')
    if not file or file.filename == '':
        flash('Pilih file CSV terlebih dahulu.', 'error')
        return redirect(url_for('dashboard'))

    if not file.filename.lower().endswith('.csv'):
        flash('File harus berformat .csv', 'error')
        return redirect(url_for('dashboard'))

    try:
        core.train_pipeline(file.stream, progress_cb=lambda msg: print(f'[TRAINING] {msg}', flush=True))
        flash('Dataset berhasil diproses dan model berhasil dilatih!', 'success')
    except ValueError as e:
        flash(str(e), 'error')
    except Exception as e:
        flash(f'Terjadi kesalahan saat memproses dataset: {e}', 'error')

    return redirect(url_for('dashboard'))


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if not core.has_trained_model():
        return jsonify({'error': 'Belum ada model. Upload dataset terlebih dahulu.'}), 400

    data = request.get_json()
    raw_text = (data.get('text') or '').strip()
    if not raw_text:
        return jsonify({'error': 'Teks ulasan tidak boleh kosong.'}), 400

    try:
        result = core.predict_sentiment(raw_text)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


# ---------------------------------------------------------------
# Laporan PDF (4 laporan terpisah)
# ---------------------------------------------------------------
REPORT_BUILDERS = {
    'dataset': ('build_report_dataset', 'Laporan_1_Dataset_Ramayana.pdf'),
    'evaluasi': ('build_report_evaluasi', 'Laporan_2_Evaluasi_Model_Ramayana.pdf'),
    'confusion': ('build_report_confusion', 'Laporan_3_Confusion_Matrix_Ramayana.pdf'),
    'kata': ('build_report_kata', 'Laporan_4_Analisis_Kata_Ramayana.pdf'),
}


@app.route('/laporan/<jenis>')
@login_required
def laporan(jenis):
    if jenis not in REPORT_BUILDERS:
        flash('Jenis laporan tidak dikenali.', 'error')
        return redirect(url_for('dashboard'))

    if not core.has_trained_model():
        flash('Belum ada hasil untuk dilaporkan. Upload dataset dulu.', 'error')
        return redirect(url_for('dashboard'))

    import report_pdf
    builder_name, download_name = REPORT_BUILDERS[jenis]
    builder_func = getattr(report_pdf, builder_name)
    metrics = core.load_metrics()
    pdf_path = builder_func(metrics, penyusun=session.get('username', '-'))
    return send_file(pdf_path, as_attachment=True, download_name=download_name)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
