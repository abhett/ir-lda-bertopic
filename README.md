# 🧠 IR-LDA-BERTopic: Analisis Topik pada Puisi

Aplikasi web berbasis Flask yang digunakan untuk menganalisis topik dari kumpulan puisi berbahasa Indonesia menggunakan 3 pendekatan:

- 🔍 **Information Retrieval (IR)** – pencarian kemiripan dokumen berbasis TF-IDF.
- 📚 **Latent Dirichlet Allocation (LDA)** – pembentukan topik berdasarkan distribusi kata.
- 🤖 **BERTopic** – pembentukan topik menggunakan embedding + clustering modern (UMAP + HDBSCAN atau KMeans).

---

## ⚙️ Teknologi yang Digunakan

- Python 3.x
- Flask
- BERTopic + Sentence Transformers
- scikit-learn
- UMAP-learn
- HDBSCAN
- NLTK + Sastrawi (Bahasa Indonesia)

---



## 🚀 Cara Menjalankan

1. **Clone repo ini**
   ```bash
   git clone https://github.com/username/ir-lda-bertopic-clean.git
   cd ir-lda-bertopic-clean

**2. Buat virtual environment**
python -m venv venv
venv\Scripts\activate  # (Windows)

**3. Install dependencies**
   pip install -r requirements.txt

**4. Jalankan aplikasi**
   python app.py

**5. Akses di browser**
      http://127.0.0.1:5000

**Dataset
asset/pelangipuisi.jsonl — dataset berisi puisi-puisi pendek, satu JSON per baris:
**


