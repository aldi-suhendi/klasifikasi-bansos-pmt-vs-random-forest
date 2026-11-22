# Klasifikasi Kelayakan Bantuan Sosial: PMT (Regresi Logistik) vs Random Forest

Aplikasi desktop dan pipeline *machine learning* untuk membandingkan dua metode klasifikasi kelayakan penerima bantuan sosial tingkat kelurahan: **Proxy Means Test (PMT) berbasis Regresi Logistik** dan **Random Forest**.

Proyek ini adalah hasil skripsi S1 Teknik Informatika, dikembangkan dengan Python, scikit-learn, dan PyQt5.

> \*\*Catatan penting tentang data.\*\* Data penelitian asli berisi informasi rumah tangga penerima bantuan, sehingga \*\*tidak dipublikasikan\*\* di repositori ini. Folder `data/` hanya berisi \*\*data dummy sintetis\*\* (`Data-Dummy.xlsx`) yang dibuat oleh `generate\_dummy\_data.py` dengan struktur kolom yang sama. Seluruh angka pada bagian "Demo pada data dummy" berasal dari data sintetis tersebut, bukan dari data warga.

## Hasil pada data asli skripsi

Ringkasan hasil dari Bab IV skripsi (2.018 rumah tangga, 607 berlabel LAYAK dan 1.411 TIDAK LAYAK; 1.614 data latih dan 404 data uji). Data asli tidak disertakan.

|Metrik (data uji)|PMT Regresi Logistik|Random Forest|
|-|-:|-:|
|Accuracy|89,60%|91,34%|
|Balanced Accuracy|88,83%|90,07%|
|Precision|80,30%|84,80%|
|Recall|86,89%|86,89%|
|F1-Score|83,46%|85,83%|
|Inclusion Error|19,70%|15,20%|
|Exclusion Error|13,11%|13,11%|
|ROC-AUC|96,39%|95,94%|

Random Forest unggul tipis pada accuracy, precision, F1, dan kesalahan inklusi, tetapi uji **Exact McNemar** (p = 0,265) menunjukkan perbedaan keduanya **belum signifikan secara statistik** pada data uji ini.

## Demo pada data dummy

Hasil menjalankan `python train.py` pada `data/Data-Dummy.xlsx` (2.000 baris sintetis, ±30% LAYAK). Angka dapat sedikit berbeda antar versi pustaka, dan **tidak boleh dibandingkan dengan hasil skripsi** karena data dibuat dengan aturan buatan (pola lebih linear, sehingga PMT wajar unggul).

|Metrik (data uji)|PMT Regresi Logistik|Random Forest|
|-|-:|-:|
|Accuracy|88,00%|86,25%|
|Balanced Accuracy|88,10%|84,23%|
|Precision|75,71%|76,00%|
|Recall|88,33%|79,17%|
|F1-Score|81,54%|77,55%|
|ROC-AUC|95,09%|93,04%|

|Perbandingan metrik|Kurva ROC|
|-|-|
|!\[Perbandingan metrik](docs/images/perbandingan-metrik.png)|!\[Kurva ROC](docs/images/kurva-roc.png)|

|Confusion matrix PMT|Confusion matrix Random Forest|
|-|-|
|!\[Confusion matrix PMT](docs/images/confusion-matrix-pmt.png)|!\[Confusion matrix RF](docs/images/confusion-matrix-rf.png)|

Gambar lain (EDA dan *feature importance*) ada di folder [`docs/images`](docs/images).



## Fitur

* **Pipeline penelitian**: EDA, pembagian data latih-uji terstratifikasi (80/20), validasi silang 5-fold, *grid search*, pemilihan cut-off, evaluasi, dan uji Exact McNemar.
* **Aplikasi desktop (PyQt5)** dengan login operator dan menu: Dashboard, Import Data, Data Warga, Hasil Klasifikasi, Prediksi Data Baru (satu warga atau batch dari Excel), Evaluasi, Laporan (ekspor Excel), dan Tentang.
* **Prediksi lewat terminal** (`prediksi\_batch.py`) tanpa membuka aplikasi.
* **Pemeriksaan integritas**: hash SHA-256 untuk file model, sidik jari (*fingerprint*) dataset acuan, dan validasi versi lingkungan agar hasil dapat direproduksi.
* **Autentikasi lokal**: kata sandi disimpan sebagai hash PBKDF2-HMAC-SHA256 dengan kebijakan kata sandi kuat.

## Metodologi singkat

* **Target**: `KETERANGAN` (LAYAK / TIDAK LAYAK).
* **Prediktor**: 27 variabel (usia dihitung dari tanggal lahir per 31 Desember 2025, kondisi rumah, fasilitas, kepemilikan aset, pekerjaan, pendidikan, dan lainnya). Kolom identitas (nama, alamat, kecamatan, kelurahan) tidak dipakai sebagai prediktor.
* **Praproses** (di dalam *pipeline*, untuk mencegah kebocoran data): imputasi median/modus, *one-hot encoding*, dan standardisasi pada PMT.
* **Model**: Regresi Logistik dan Random Forest dengan `class\_weight="balanced"`, dioptimasi dengan `GridSearchCV` (skor F1).
* **Cut-off**: dipilih dari kurva ambang 0,05–0,95 yang dihitung pada data latih.
* **Evaluasi**: accuracy, balanced accuracy, precision, recall, F1, *inclusion/exclusion error*, ROC-AUC, average precision, dan Exact McNemar.

Konfigurasi terpusat ada di [`src/config.py`](src/config.py).

## Struktur proyek

```
.
├── aplikasi-skripsi.py        # Aplikasi desktop (PyQt5)
├── train.py                   # Pelatihan + evaluasi + gambar + laporan
├── prediksi\_batch.py          # Prediksi data baru lewat terminal
├── eda.py                     # Gambar EDA
├── generate\_dummy\_data.py     # Pembuat data dummy sintetis
├── setup\_akun\_demo.py         # Membuat akun operator lokal
├── buat\_akun\_operator.py      # Ganti akun operator
├── ubah\_password\_operator.py  # Ganti kata sandi
├── cek\_dataset.py / cek\_lingkungan.py
├── src/                       # Modul inti (data, modeling, evaluasi, auth, dll.)
├── data/Data-Dummy.xlsx       # Data sintetis
├── template/                  # Template Excel untuk prediksi data baru
├── docs/images/               # Gambar hasil demo
└── requirements.txt
```

## Cara menjalankan

Disarankan **Python 3.13** dan versi pustaka pada `requirements.txt` (dikunci agar hasil Random Forest dapat direproduksi).

**Windows (PowerShell)**

```powershell
git clone https://github.com/ALDI SUHENDI/klasifikasi-bansos-pmt-vs-random-forest.git
cd klasifikasi-bansos-pmt-vs-random-forest

py -3.13 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt

python train.py                 # melatih model pada data dummy (hasil masuk ke folder output/)
python setup\_akun\_demo.py       # membuat akun operator (sekali saja)
python aplikasi-skripsi.py      # membuka aplikasi
```

**macOS / Linux**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python setup\_akun\_demo.py
python aplikasi-skripsi.py
```

Untuk mengimpor dataset di aplikasi, pilih `data/Data-Dummy.xlsx`. Untuk mencoba prediksi data baru, unduh template dari menu **Prediksi Data Baru** (atau lihat folder `template/`).

Opsional:

```bash
python generate\_dummy\_data.py --rows 2000 --seed 2026   # buat ulang data dummy
python eda.py                                           # gambar EDA
python prediksi\_batch.py data\_baru.xlsx                 # prediksi lewat terminal
```

### Memakai versi Python/pustaka yang berbeda

Secara bawaan program berhenti jika versi lingkungan tidak sama dengan yang dikunci. Untuk sekadar mencoba demo di versi lain, aktifkan mode longgar (hanya memberi peringatan):

```powershell
$env:APK\_RELAXED\_ENV = "1"      # PowerShell
```

```bash
export APK\_RELAXED\_ENV=1        # macOS / Linux
```

Hasil Random Forest dapat sedikit berbeda pada versi pustaka lain.

## Keterbatasan

* Purwarupa lokal untuk penelitian, **bukan sistem produksi** dan tidak untuk dipakai sebagai satu-satunya dasar keputusan penyaluran bantuan.
* Model dilatih pada satu kelurahan; validasi wilayah pada aplikasi masih khusus Kelurahan Pakojan, Kecamatan Pinang. Data dummy menggunakan label wilayah tersebut agar alur aplikasi dapat dicoba, namun isinya sintetis.
* Akun operator bersifat lokal dan sederhana (tanpa manajemen peran/multi-pengguna).

## Penulis

**Aldi Suhendi**, S1 Teknik Informatika, Universitas Islam Syekh Yusuf (2026).
Kontak: *aldi271200@gmail.com / linkedin.com/in/aldi-suhendi-35ab691b6*

## Lisensi

[MIT](LICENSE) untuk kode. Data pada `data/Data-Dummy.xlsx` adalah data sintetis dan bebas dipakai.

