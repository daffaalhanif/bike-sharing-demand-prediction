# 🚲 Bike Sharing Demand Prediction

## 📌 Deskripsi Project

Project ini merupakan implementasi machine learning end-to-end untuk
memprediksi jumlah penyewaan sepeda per jam pada sistem Capital Bikeshare
di Washington D.C. Model dibangun untuk membantu operator merencanakan
distribusi sepeda secara proaktif dan data-driven, menggantikan pendekatan
reaktif berbasis intuisi yang rawan terhadap kondisi under-supply maupun
over-supply.

## ❗️ Problem Statement

Operator bike-sharing menghadapi tantangan ketidakpastian demand yang
berfluktuasi secara dinamis setiap jamnya. Tanpa sistem prediksi yang andal,
operator terpaksa mengandalkan distribusi manual yang rawan terhadap dua
risiko utama:

- **Under-supply** - calon penyewa tidak terlayani, meningkatkan churn rate
- **Over-supply** - sepeda menumpuk di stasiun tertentu, meningkatnya biaya
  redistribusi

## 🎯 Tujuan

Membangun model machine learning berbasis regresi yang mampu memprediksi
jumlah total penyewaan sepeda (`cnt`) per jam secara akurat berdasarkan
fitur temporal dan kondisi cuaca, sehingga operator dapat merencanakan
distribusi sepeda secara proaktif.

**Target kuantitatif:**
- RMSE minimal 50% lebih baik dari basline Linear Regression
- MAE ≤ 87.63 sepeda (50% lebih baik dari naive baseline)
- R² di atas 0.85

## 🔍 Pendekatan Analitik

- **Jenis masalah:** Supervised Learning - Regresi
- **Target variabel:** `cnt` (total penyewaan sepeda per jam)
- **Evaluation metrics:** RMSE sebagai metrik utama karena memberikan
  penalti lebih besar pada error besar - sesuai konteks operasional
  di mana kesalahan prediksi skala besar berdampak langsung pada
  ketersediaan sepeda. MAE digunakan sebagai acuan operasional karena
  interpretasinya intuitif dalam satuan sepeda. R² digunakan sebagai
  informasi tambahan seberapa baik model menangkap pola keseluruhan data.

## 📊 Dataset

Dataset berisi data penyewaan sepeda per jam dari Capital Bikeshare,
Washington D.C., periode 2011-2012 dengan 12.165 baris dan 11 kolom.

**Sumber:** [Capital Bikeshare System Data](http://capitalbikeshare.com/system-data)

## 🤖 Model dan Hasil

Enam algoritma diuji dan dibandingkan - dari baseline Linear Regression
hingga ensembel model. LightGBM terpilih sebagai final model setelah
proses hyperparameter tuning dua fase (RandomizedSearch + BayesSearch).

**Cara kerja LightGBM:** Model membangun sekumpulan pohon keputusan
secara bertahap - tiap pohon baru difokuskan untuk memperbaiki error
dari pohon sebelumnya. LightGBM menggunakan pendekatan leaf-wise growth
yang lebih efisien, sehingga mampu menangkap pola non-linear kompleks
seperti pola bimodal rush hour yang tidak bisa ditangkap Linear Regression.

| Metric | Target | Hasil | Status |
|---|---|---|---|
| RMSE | 50% lebih baik dari LR baseline (164.13) | 64.50 (↓ 61%) | ✅ |
| MAE | ≤ 87.63 sepeda | 43.91 (↓ 75% dari naive baseline) | ✅ |
| R² | > 0.85 | 0.9144 | ✅ |

## 📝 Kesimpulan

Ketiga target berhasil dipenuhi. Model LightGBM yang dibangun mampu
memprediksi demand sepeda per jam dengan rata-rata selisih 44 sepeda
dari nilai aktual — jauh lebih baik dari pendekatan tanpa model yang
meleset rata-rata 175 sepeda per jam. Model paling andal digunakan
pada kondisi cuaca cerah hingga hujan ringan (weathersit 1-3) di
seluruh segmen waktu operasional.

## 💡 Rekomendasi

- Model paling tepat digunakan sebagai alat bantu perencanaan
  distribusi sepeda sebelum jam operasional dimulai
- Tambahkan buffer stok di jam rush hour pagi (06-09) karena
  MAE tertinggi ada di segmen ini (47.69 sepeda)
- Model perlu di-retrain secara berkala dengan data terbaru
  seiring pertumbuhan pengguna
- Integrasikan forecast cuaca per jam dari API cuaca eksternal
  untuk mengotomasi pipeline prediksi operasional

## ⚠️ Limitasi Model

- Dataset hanya mencakup 2011-2012 — model perlu di-retrain
  secara berkala dengan data terbaru
- Prediksi untuk kondisi cuaca ekstrem (`weathersit = 4`) tidak
  dapat diandalkan karena hanya ada 3 baris data di training set
- Model memprediksi total demand di tingkat sistem, bukan per stasiun

## 🛠️ Tech Stack

**Data Manipulation & Analysis:**
- `pandas` — manipulasi DataFrame dan agregasi data
- `numpy` — operasi array dan kalkulasi numerik

**Visualization:**
- `matplotlib` — layout canvas dan kustomisasi chart
- `seaborn` — visualisasi statistik utama

**Machine Learning:**
- `scikit-learn` — preprocessing (StandardScaler, Pipeline),
  cross-validation (TimeSeriesSplit), hyperparameter tuning
  (RandomizedSearchCV), evaluation metrics
- `scikit-optimize` — Bayesian hyperparameter tuning (BayesSearchCV)
- `lightgbm` — final model
- `xgboost` — model kandidat tuning

## 📁 Struktur Repository

- `bike_sharing_demand_prediction.ipynb` — Notebook dokumentasi lengkap
- `bike_sharing_demand_lgbm.pkl` — Final model (LightGBM Tuned)
- `data_bike_sharing.csv` — Dataset
- `README.md` — Project introduction
