# Structural Damage Classification — Gempa Gorkha, Nepal 2015

Prediksi tingkat kerusakan bangunan (`damage_grade`, Grade 1–5) dari karakteristik struktural dan lokasi,
menggunakan data survei pasca-gempa Gorkha (Mw 7,8; 25 April 2015).
Tugas 1 *Supervised Machine Learning* — kompetisi Kaggle
[Structural Damage Classification](https://www.kaggle.com/competitions/structural-damage-classification).

| | |
|---|---|
| **Tipe masalah** | Klasifikasi multikelas (5 kelas, tidak seimbang) |
| **Metrik** | Macro F1-Score |
| **Model** | `Pipeline` scikit-learn: feature engineering custom + `ColumnTransformer` + LightGBM |
| **Validasi** | Stratified 5-fold CV, seluruh pipeline di-fit ulang tiap fold |
| **CV macro F1** | ≈ 0,596 (iterasi sebelumnya, lihat [Hasil](#hasil)) |

---

## Struktur repository

```
structural-damage-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                     # dataset Kaggle (tidak di-commit)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_eda.ipynb          # eksplorasi data & hipotesis
│   ├── 02_modeling.ipynb     # pipeline, tuning, CV, model final
│   └── 03_submission.ipynb   # prediksi test & file submission
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py         # path, load data, kolom, ColumnTransformer
│   ├── feature_engineering.py# transformer custom (sklearn API)
│   ├── train.py              # pipeline, tuning, CV, bobot kelas, training final
│   └── predict.py            # load model & tulis submission
├── models/
│   └── pipeline_model_lgbm.pkl
├── submissions/
│   └── submission.csv
└── reports/
    ├── cv_summary.json
    ├── tuning_results.csv
    └── figures/
        ├── feature_importance.png
        ├── confusion_matrix.png
        └── ... (gambar EDA)
```

---

## Cara mengunduh data

**Opsi A — Website Kaggle:** buka halaman kompetisi → *Join Competition* → *I Understand and Accept* →
tab **Data** → **Download All**. Ekstrak `train.csv`, `test.csv`, `sample_submission.csv` ke folder `data/`.

**Opsi B — Kaggle API:**

```bash
pip install kaggle
# simpan API token di ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token)
kaggle competitions download -c structural-damage-classification -p data
unzip data/structural-damage-classification.zip -d data
```

> Aturan kompetisi harus sudah di-*accept* di website, kalau tidak download/submit via API akan gagal.

---

## Cara menjalankan

### Lokal

```bash
git clone https://github.com/<username>/structural-damage-classification.git
cd structural-damage-classification
pip install -r requirements.txt
# letakkan file CSV di data/
```

Lewat notebook (berurutan): `01_eda.ipynb` → `02_modeling.ipynb` → `03_submission.ipynb`.

Lewat command line:

```bash
python -m scripts.train --tune        # tuning + CV + model final -> models/, reports/
python -m scripts.train               # tanpa tuning (parameter default)
python -m scripts.predict             # -> submissions/submission.csv
```

### Kaggle Notebook

1. Buat notebook baru dari halaman kompetisi (data otomatis ada di `/kaggle/input/...`).
2. Pilih salah satu cara membawa kode:
   - **Clone dari GitHub** (Settings → Internet **On**, perlu verifikasi nomor HP): isi `REPO_URL` di cell
     *Setup* notebook, atau jalankan
     ```python
     !git clone --depth 1 https://github.com/<username>/structural-damage-classification.git /kaggle/working/repo
     %cd /kaggle/working/repo
     !python -m scripts.train --tune
     !python -m scripts.predict
     ```
   - **Upload sebagai Dataset:** upload folder repo (minimal `scripts/`) sebagai Kaggle Dataset lalu
     tambahkan ke notebook lewat *Add Input*; cell *Setup* mencarinya otomatis di `/kaggle/input`.
3. Output (model, gambar, submission) ditulis ke `/kaggle/working`, dan `submission.csv` juga disalin ke
   `/kaggle/working/submission.csv` agar bisa langsung di-submit.

Estimasi waktu di CPU Kaggle: tuning ±20–30 menit, CV 5-fold ±30–40 menit, model final ±8 menit.

---

## Workflow analisis

1. **EDA** (`01_eda.ipynb`): kualitas data, distribusi target, hubungan fitur–kerusakan, analisis
   granularitas lokasi dan tetangga ID.
2. **Feature engineering** (`scripts/feature_engineering.py`), semuanya transformer scikit-learn:
   | Transformer | Isi | Supervised? |
   |---|---|---|
   | `StructuralFeatures` | prefiks `building_id` (kode distrik/VDC/ward lama), periode alami `0,075·H^0,75`, kelangsingan, luas total, umur×lantai, peringkat kerentanan material (ala EMS-98) | tidak |
   | `GroupRelativeFeatures` | ukuran area, selisih umur/tinggi/lantai/luas/kerentanan terhadap rata-rata area, pangsa material di area | tidak |
   | `HierarchicalTargetEncoder` | probabilitas tiap grade per wilayah, di-*smoothing* ke wilayah induk (distrik → VDC → ward → blok), plus kombinasi wilayah × material/fondasi/atap/lantai | ya, out-of-fold |
   | `IdNeighborEncoder` | grade bangunan dengan nomor ID terdekat di ward lama yang sama (k = 3, 10, 30) | ya, out-of-fold |
   | `CategoricalCoder` | kategori konstruksi → `pandas.Categorical` (split native LightGBM) | tidak |
3. **Pipeline** (`scripts/train.py → build_pipeline`):
   `StructuralFeatures → ColumnTransformer(num | cat | grp | te | nb) → LGBMClassifier`.
4. **Pencegahan kebocoran:** encoder berlabel memakai *cross-fitting* di `fit_transform`
   (setiap baris train di-encode dari fold lain). Data baru di-encode dengan rata-rata tabel per fold,
   sehingga distribusinya sama dengan nilai out-of-fold saat training. Karena seluruh pipeline di-fit ulang
   di tiap fold CV dan tiap kandidat tuning, data validasi tidak pernah ikut membentuk fiturnya sendiri.
5. **Tuning:** `RandomizedSearchCV` pada pipeline (scoring `f1_macro`), dijalankan pada subsampel
   wilayah utuh dengan cache transformer; parameter default dipertahankan bila lebih baik.
6. **Evaluasi:** Stratified 5-fold out-of-fold → macro F1 per fold, classification report, confusion matrix.
7. **Optimasi macro F1:** probabilitas dikalikan bobot per kelas (coordinate search di OOF); diuji jujur
   dengan bobot dari 4 fold yang dievaluasi di fold ke-5. Bobot disimpan di dalam model
   (`WeightedArgmaxClassifier`).
8. **Submission** (`03_submission.ipynb` / `scripts/predict.py`): format & urutan sama dengan
   `sample_submission.csv`.

---

## Hasil

> Angka di bawah dari **iterasi sebelumnya** (pipeline setara berbasis skrip). Perbarui dengan
> `reports/cv_summary.json` setelah menjalankan `02_modeling.ipynb`.

| Eksperimen | CV macro F1 |
|---|---|
| Baseline: tebak grade terbanyak per `ward_id` | — (akurasi 0,528) |
| LightGBM, fitur mentah + ID lokasi kategorikal | 0,533 |
| + prefiks `building_id`, target encoding berjenjang, tetangga ID, fitur area | 0,591 |
| + bobot kelas untuk macro F1 (held-out) | **0,596** |
| Iterasi ini (pipeline + tuning) | _isi dari `cv_summary.json`_ |

| Leaderboard | Skor |
|---|---|
| Public | _isi_ |
| Private | _diumumkan setelah kompetisi selesai_ |

F1 per kelas (iterasi sebelumnya): Grade 1 = 0,69 · Grade 2 = 0,47 · Grade 3 = 0,49 · Grade 4 = 0,57 · Grade 5 = 0,76.

---

## Interpretasi

**1. Lokasi adalah faktor dominan karena menggantikan data guncangan yang tidak ada.**
Dataset tidak memuat percepatan tanah, jarak ke episentrum, atau jenis tanah secara detail. Intensitas
guncangan bervariasi besar antar wilayah, sehingga model belajar "seberapa keras guncangan di sini"
dari kerusakan bangunan lain di wilayah yang sama. Fitur terpenting adalah target encoding pada level
wilayah paling halus (prefiks 10 digit `building_id`) dan label tetangga ID. Kode wilayah lama di
`building_id` lebih informatif daripada `ward_id` baru karena unitnya lebih kecil.

**2. Konstruksi menentukan kerentanan, dan interaksinya dengan lokasi penting.**
Kombinasi wilayah × tipe fondasi dan wilayah × material struktur termasuk fitur teratas: di wilayah yang
sama, bangunan batu-lumpur rusak jauh lebih parah daripada bangunan semen-bata atau beton bertulang.
Ini sesuai dengan konsep kurva kerentanan (*fragility*): kerusakan = f(intensitas guncangan, kerentanan bangunan).

**3. Proksi dinamika struktur membantu, tetapi kontribusinya kecil.**
Periode alami, kelangsingan, dan kerentanan relatif terhadap area (`vuln_rel_id6`) ikut dipakai model,
namun informasinya terbatas karena data hanya berisi tinggi, lantai, dan luas, tanpa detail elemen struktur.

**4. Grade tengah paling sulit.**
Grade 2 dan 3 memiliki F1 terendah: grade bersifat ordinal dan penilaian surveyor di batas antar-grade
subjektif, sehingga kesalahan paling banyak terjadi ke grade tetangga (lihat `confusion_matrix.png`).
Optimasi bobot kelas sedikit mengorbankan Grade 5 untuk menaikkan recall Grade 2–4, dan itu menaikkan macro F1.

**Keterbatasan:** tidak ada koordinat geografis maupun data seismik; label adalah penilaian visual
surveyor. Model Neural Network (termasuk GNN) tidak dipakai sesuai aturan tugas.

---

## Aturan yang diikuti

- Dataset tidak di-commit (`data/*.csv` di `.gitignore`).
- Preprocessing, training, dan tuning memakai `Pipeline` + `ColumnTransformer`.
- Setiap function/class di `scripts/*.py` memiliki docstring.
- Tidak menggunakan model Neural Network.
