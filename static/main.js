// ---------- Upload dataset (halaman Dashboard) ----------
const fileInput = document.getElementById('dataset_file');
const fileLabel = document.getElementById('file-label');
const uploadForm = document.getElementById('upload-form');
const uploadBtn = document.getElementById('upload-btn');
const uploadLoading = document.getElementById('upload-loading');

if (fileInput) {
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
        }
    });
}

if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Memproses...';
        uploadLoading.classList.remove('hidden');
    });
}

// ---------- Prediksi (halaman Prediksi) ----------
const input = document.getElementById('review-input');
const charCount = document.getElementById('char-count');
const btn = document.getElementById('predict-btn');
const resultArea = document.getElementById('result-area');
const errorArea = document.getElementById('error-area');
const COLORS = { Positif: '#4C7A51', Netral: '#C9A227', Negatif: '#8B1E3F' };

if (input) {
    input.addEventListener('input', () => {
        charCount.textContent = `${input.value.length} karakter`;
    });
}

if (btn) {
    btn.addEventListener('click', async () => {
        const text = input.value.trim();
        errorArea.classList.add('hidden');
        resultArea.classList.add('hidden');

        if (!text) {
            errorArea.textContent = 'Masukkan teks ulasan terlebih dahulu.';
            errorArea.classList.remove('hidden');
            return;
        }

        btn.disabled = true;
        btn.textContent = 'Menganalisis...';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            const data = await res.json();

            if (!res.ok) {
                errorArea.textContent = data.error || 'Terjadi kesalahan.';
                errorArea.classList.remove('hidden');
                return;
            }

            const tag = document.getElementById('result-tag');
            tag.textContent = data.label;
            tag.className = `result-tag ${data.label}`;
            document.getElementById('result-confidence').textContent =
                `Keyakinan model: ${data.proba[data.label]}%`;

            const barsContainer = document.getElementById('proba-bars');
            barsContainer.innerHTML = '';
            ['Positif', 'Netral', 'Negatif'].forEach((label) => {
                if (!(label in data.proba)) return;
                const pct = data.proba[label];
                const row = document.createElement('div');
                row.className = 'proba-row';
                row.innerHTML = `
                    <span>${label}</span>
                    <div class="proba-track"><div class="proba-fill" style="width:${pct}%; background:${COLORS[label]}"></div></div>
                    <span>${pct}%</span>
                `;
                barsContainer.appendChild(row);
            });

            document.getElementById('clean-text-output').textContent = data.clean_text || '(kosong)';
            resultArea.classList.remove('hidden');
        } catch (err) {
            errorArea.textContent = 'Gagal terhubung ke server.';
            errorArea.classList.remove('hidden');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Analisis Sentimen';
        }
    });
}

// ---------- Tabs (halaman Dashboard) ----------
document.querySelectorAll('.tab-btn').forEach((tabBtn) => {
    tabBtn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
        tabBtn.classList.add('active');
        document.getElementById(`tab-${tabBtn.dataset.tab}`).classList.add('active');
    });
});
