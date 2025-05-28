# Laporan Proyek Machine Learning - Septiadi Bayu Eka Samudera Istiyono

## Domain Proyek
Domain proyek ini adalah predictive analytics dalam sektor biaya layanan kesehatan. Analisis prediktif menggunakan berbagai teknik statistik dan machine learning untuk membuat prediksi tentang hasil di masa depan berdasarkan data historis (Siegel, 2016; Beg et al., 2024). Dalam konteks biaya kesehatan, tujuannya adalah untuk mengidentifikasi faktor-faktor yang mempengaruhi pengeluaran medis individu dan membangun model yang dapat mengestimasi biaya tersebut. Aplikasi dari analisis ini sangat luas, mulai dari membantu perusahaan asuransi dalam penetapan premi yang akurat, hingga membantu individu dalam merencanakan keuangan kesehatan mereka. Proyek ini secara spesifik akan fokus pada prediksi biaya medis pribadi menggunakan atribut demografis dan kesehatan.

### Latar Belakang
Biaya layanan kesehatan terus meningkat secara global, menjadi salah satu tantangan signifikan bagi individu, penyedia layanan kesehatan, dan perusahaan asuransi. Menurut Organisasi Kesehatan Dunia (WHO), belanja kesehatan global terus meningkat, dan pembiayaan yang tidak efisien dapat menghambat akses ke layanan yang dibutuhkan (WHO, n.d.). Kemampuan untuk memprediksi biaya medis menjadi krusial dalam upaya mengelola dan mengendalikan pengeluaran ini. Prediksi biaya medis yang akurat memungkinkan perencanaan anggaran yang lebih baik bagi individu dan keluarga, membantu penyedia layanan kesehatan dalam alokasi sumber daya yang optimal, serta mendukung perusahaan asuransi dalam menetapkan premi yang adil dan kompetitif (Koonin et al., 2020; Alam et al., 2024).
Faktor-faktor yang memengaruhi biaya medis sangat beragam, meliputi aspek demografis seperti usia dan jenis kelamin, pilihan gaya hidup seperti merokok, kondisi kesehatan yang direfleksikan oleh Indeks Massa Tubuh (BMI), serta faktor regional. Menganalisis hubungan antara faktor-faktor ini dengan biaya medis dapat memberikan wawasan berharga. Sebagai contoh, penelitian telah secara konsisten menunjukkan bahwa status merokok merupakan prediktor kuat untuk biaya kesehatan yang lebih tinggi (Lightwood & Glantz, 2016). Demikian pula, usia dan BMI seringkali berkorelasi positif dengan peningkatan kebutuhan layanan kesehatan dan, akibatnya, biaya yang lebih tinggi.
Dalam proyek ini akan dibuat beberapa model Machine Learning yang kemudian dievaluasi untuk membandingkan model mana yang hasil prediksinya paling baik lalu diharapkan dapat memprediksi biaya medis pribadi berdasarkan atribut-atribut pasien dengan memanfaatkan dataset `insurance.csv`.

### Tujuan Proyek
Memahami faktor-faktor yang memengaruhi biaya medis pribadi sangat penting dalam berbagai aspek industri kesehatan dan perencanaan keuangan individu. Perusahaan asuransi dapat menggunakan pemahaman ini untuk menentukan premi yang lebih akurat dan adil bagi nasabahnya. Penyedia layanan kesehatan dapat mengoptimalkan alokasi sumber daya mereka berdasarkan prediksi kebutuhan medis. Bagi individu, kemampuan untuk memprediksi biaya medis membantu dalam perencanaan keuangan kesehatan yang lebih baik dan pengambilan keputusan terkait pilihan asuransi atau gaya hidup.

Proyek ini bertujuan untuk membangun model prediktif yang dapat mengestimasi biaya medis berdasarkan atribut-atribut pasien. Dataset yang digunakan adalah `insurance.csv`, yang berisi informasi demografis dan kesehatan pasien. Dengan menganalisis data ini, kita dapat mengidentifikasi faktor-faktor kunci yang paling berpengaruh terhadap biaya medis dan mengembangkan alat prediksi yang berguna.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, rincian masalahnya adalah sebagai berikut:
-   Bagaimana cara memprediksi biaya medis pribadi berdasarkan atribut demografis (usia, jenis kelamin, wilayah) dan atribut kesehatan (indeks massa tubuh/BMI, jumlah anak, status perokok) individu?
-   Faktor-faktor apa saja yang paling signifikan dalam mempengaruhi variasi biaya medis pribadi antar individu?
-   Model machine learning manakah yang memberikan performa terbaik dalam memprediksi biaya medis dengan tingkat kesalahan (error) yang dapat diterima?

### Goals

Untuk menjawab pertanyaan di atas, maka akan dijabarkan sebagai berikut:
-   Mengembangkan sebuah model prediktif machine learning yang mampu mengestimasi biaya medis pribadi dengan akurasi yang baik.
-   Mengidentifikasi dan menganalisis fitur-fitur kunci yang memiliki kontribusi paling besar terhadap perbedaan biaya medis.
-   Mengevaluasi beberapa algoritma machine learning regresi dan memilih model yang paling sesuai dan memberikan performa terbaik untuk kasus prediksi biaya medis ini, berdasarkan metrik evaluasi Mean Absolute Error (MAE).

### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya sebagai berikut:
-   Membuat dan mengevaluasi 5 model Machine Learning yaitu dengan algoritma Linear Regression, K-Nearest Neighbors (KNN) Regressor, Support Vector Regression (SVR), Random Forest Regressor, dan Gradient Boosting Regressor.
-   Melakukan pra-pemrosesan data yang komprehensif untuk meningkatkan kualitas data dan performa model, termasuk penanganan data duplikat, transformasi log pada variabel target, *one-hot encoding* fitur kategorikal, dan standarisasi fitur numerik.
-   Perkiraan biaya medis adalah tujuan yang ingin dicapai. Ini merupakan permasalahan regresi. Untuk kasus regresi seperti ini akan digunakan metrik Mean Absolute Error (MAE) dan Mean Squared Error (MSE). Metrik ini mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. Nantinya masing-masing model akan dievaluasi untuk memilih algoritma dengan nilai metrik terbaik.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah "Medical Cost Personal Datasets" yang diperoleh dari Kaggle. Sumber dataset: [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance). Dataset ini berisi 1338 entri (setelah penghapusan satu data duplikat menjadi 1337 entri) dan 7 kolom fitur.

### Variabel-variabel pada dataset `insurance.csv` adalah sebagai berikut:
* **age**: Usia penerima manfaat utama (numerik).
* **sex**: Jenis kelamin kontraktor asuransi (kategorikal: female, male).
* **bmi**: Indeks massa tubuh (numerik).
* **children**: Jumlah anak yang ditanggung (numerik).
* **smoker**: Status apakah merokok atau tidak (kategorikal: yes, no).
* **region**: Wilayah tempat tinggal (kategorikal: northeast, southeast, southwest, northwest).
* **charges**: Biaya medis individu (target, numerik).

### Exploratory Data Analysis - Univariate Analysis
* **Informasi Dasar Dataset**: Tidak ada nilai yang hilang. Terdapat satu data duplikat yang dihapus.
    ```python
    print(df.isnull().sum())
    df = df.drop_duplicates()
    ```
* **Fitur Numerik (`age`, `bmi`, `children`, `charges`)**:
    * `charges` memiliki distribusi sangat miring ke kanan dan banyak outlier (sebelum transformasi).
    * `bmi` mendekati distribusi normal.
    * `age` relatif seragam dengan dua puncak.
    * `children` mayoritas bernilai 0.
    * Setelah transformasi log pada `charges`, distribusinya menjadi lebih simetris.
    * **Fitur Kategorikal (`sex`, `smoker`, `region`)**:
    * Jumlah laki-laki dan perempuan hampir seimbang.
    * Mayoritas non-perokok.
    * Wilayah `southeast` memiliki data terbanyak.
    ### Exploratory Data Analysis - Multivariate Analysis
* **Korelasi Fitur Numerik**:
    Matriks korelasi menunjukkan korelasi positif sedang antara `age` dan `charges` (0.30 sebelum transformasi log).
    * **Fitur Kategorikal vs. Charges**:
    Boxplot menunjukkan perokok (`smoker_yes`) memiliki biaya medis jauh lebih tinggi. Wilayah (`region`) tidak menunjukkan perbedaan biaya yang sangat signifikan.
    ## Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:
-   **Penghapusan Data Duplikat**: Menghapus satu baris data duplikat.
-   **Transformasi Log pada Variabel Target (`charges`)**: Menggunakan `np.log1p` untuk menormalkan distribusi.
-   **Encoding Fitur Kategorikal**: Menggunakan `pd.get_dummies` dengan `drop_first=True` untuk fitur `sex`, `smoker`, dan `region`.
-   **Pemisahan Fitur (X) dan Target (y)**.
-   **Pembagian Data Latih dan Uji**: Rasio 80:20 menggunakan `train_test_split` (`random_state=42`).
-   **Standarisasi Fitur Numerik**: Menggunakan `StandardScaler` pada fitur `age`, `bmi`, dan `children` (di-fit pada data latih).

## Modeling
Pada tahap ini, beberapa model machine learning regresi dikembangkan untuk memprediksi biaya medis. Semua model dilatih menggunakan data latih yang telah dipersiapkan dan dievaluasi pada data uji. Target `charges` yang digunakan dalam pelatihan adalah yang telah ditransformasi log.

Model-model yang digunakan beserta penjelasannya adalah sebagai berikut:

1.  **Linear Regression**:
    * Deskripsi: Regresi Linear adalah salah satu teknik pemodelan statistik yang paling dasar dan umum digunakan untuk menjelaskan hubungan antara satu variabel dependen (target) dengan satu atau lebih variabel independen (fitur) melalui garis lurus (Montgomery et al., 2021; Močarníková & Greguš, 2019). Model ini mengasumsikan adanya hubungan linear antara fitur dan target.
    * Parameter: Menggunakan parameter default dari `sklearn.linear_model.LinearRegression`.
    * Kelebihan: Mudah diinterpretasikan, komputasi cepat, dan menjadi dasar bagi banyak teknik yang lebih kompleks.
    * Kekurangan: Mengasumsikan hubungan linear, sensitif terhadap outlier, dan mungkin tidak menangkap pola data yang kompleks.

2.  **K-Nearest Neighbors (KNN) Regressor**:
    * Deskripsi: KNN adalah algoritma non-parametrik yang digunakan untuk klasifikasi dan regresi. Dalam regresi, prediksi untuk titik data baru adalah rata-rata (atau median) dari nilai target 'k' tetangga terdekatnya di ruang fitur (Altman, 1992; Beg et al., 2024). Pemilihan 'k' dan metrik jarak sangat mempengaruhi performanya.
    * Parameter: `n_neighbors=7`.
    * Kelebihan: Sederhana untuk dipahami dan diimplementasikan, mampu menangkap hubungan non-linear tanpa membuat asumsi tentang distribusi data.
    * Kekurangan: Komputasi bisa mahal pada dataset besar (karena perlu menghitung jarak ke semua titik data latih), sensitif terhadap skala fitur dan fitur yang tidak relevan, dan performanya sangat bergantung pada pemilihan 'k' dan metrik jarak.

3.  **Support Vector Regression (SVR)**:
    * Deskripsi: SVR adalah adaptasi dari Support Vector Machines (SVM) untuk masalah regresi. Tujuannya adalah untuk menemukan fungsi (hyperplane) yang memiliki deviasi paling banyak $\epsilon$ dari target aktual untuk semua data latih, dan sekaligus se-datar mungkin (Smola & Schölkopf, 2004). SVR menggunakan kernel untuk memetakan data ke ruang dimensi yang lebih tinggi agar dapat menangani hubungan non-linear.
    * Parameter: `kernel='rbf'`, `C=1000`, `gamma=0.1`.
    * Kelebihan: Efektif di ruang dimensi tinggi, bekerja baik dengan data non-linear (dengan kernel yang tepat), dan memiliki fleksibilitas dalam pemilihan fungsi loss (tidak terlalu sensitif terhadap outlier seperti metode kuadrat terkecil).
    * Kekurangan: Membutuhkan tuning parameter yang cermat (kernel, C, gamma, epsilon), komputasi bisa intensif terutama untuk dataset besar, dan interpretasi modelnya kurang intuitif.

4.  **Random Forest Regressor**:
    * Deskripsi: Random Forest adalah metode ensemble learning yang terdiri dari banyak decision tree yang dibangun secara independen. Untuk regresi, hasil prediksi adalah rata-rata dari prediksi semua pohon individu (Breiman, 2001; Alam et al., 2024). Random Forest mengurangi varians dan overfitting yang sering terjadi pada decision tree tunggal.
    * Parameter: `n_estimators=100`, `random_state=42`, `max_depth=7`, `min_samples_leaf=5`, `min_samples_split=10`.
    * Kelebihan: Kuat terhadap overfitting (jika parameter di-tune dengan baik), mampu menangani hubungan non-linear dan interaksi antar fitur, memberikan estimasi pentingnya fitur, dan relatif mudah digunakan.
    * Kekurangan: Kurang interpretatif dibandingkan model linear atau *decision tree* tunggal, bisa lebih lambat untuk training pada data sangat besar, dan mungkin tidak bekerja dengan baik pada data dengan dimensi sangat tinggi dan jarang (sparse).

5.  **Gradient Boosting Regressor**:
    * Deskripsi: GGradient Boosting adalah teknik ensemble learning yang membangun model (biasanya decision tree) secara sekuensial. Setiap model baru dilatih untuk memperbaiki kesalahan (residual) dari model-model sebelumnya (Friedman, 2001).
    * Parameter: `n_estimators=100`, `learning_rate=0.05`, `random_state=42`, `max_depth=3`, `min_samples_leaf=10`, `min_samples_split=10`.
    * Kelebihan: Seringkali memberikan akurasi prediksi yang sangat tinggi, fleksibel dalam menangani berbagai jenis data dan fungsi loss, dan dapat menangani data yang hilang secara implisit.
    * Kekurangan: Rentan terhadap overfitting jika tidak di-tune dengan baik (terutama dengan jumlah pohon yang banyak), training bisa lebih lama karena sifatnya yang sekuensial, dan lebih kompleks untuk diinterpretasikan.

**Pemilihan Model Terbaik**:
Setelah melatih dan mengevaluasi kelima model tersebut menggunakan metrik MAE dan MSE pada data uji (dengan nilai target dikembalikan ke skala aslinya), **Random Forest Regressor** dipilih sebagai model terbaik. Alasan utama pemilihan ini adalah karena Random Forest Regressor menghasilkan **Mean Absolute Error (MAE) terendah** pada data uji, yaitu sebesar 1964.85. MAE yang lebih rendah menunjukkan bahwa rata-rata selisih absolut antara prediksi model dan nilai aktual biaya medis adalah yang paling kecil dibandingkan model lainnya, mengindikasikan performa prediksi yang lebih baik pada data yang belum pernah dilihat sebelumnya.

## Evaluation
Metrik evaluasi yang digunakan untuk menilai performa model dalam proyek ini adalah Mean Absolute Error (MAE) dan Mean Squared Error (MSE). Karena variabel target `charges` telah ditransformasi menggunakan logaritma (`np.log1p`), maka prediksi dari model juga berada dalam skala log. Untuk mendapatkan interpretasi error dalam skala biaya aslinya, prediksi dan nilai target asli (sebelum transformasi) dikembalikan ke skala semula menggunakan `np.expm1` sebelum menghitung MAE dan MSE.

$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

Keterangan:
N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi

### Perbandingan Performa Model:

| Model                       | Train MAE | Test MAE  | Train MSE     | Test MSE      |
| :-------------------------- | :-------- | :-------- | :------------ | :------------ |
| Random Forest Regressor     | 1795.02   | 1964.85   | 16824877.95   | 17779243.87   |
| Gradient Boosting Regressor | 1953.52   | 2031.02   | 19190813.56   | 19086712.12   |
| SVR                         | 1926.94   | 3313.09   | 17964739.69   | 41462956.77   |
| Linear Regression           | 4257.30   | 3755.92   | 70379311.89   | 51797278.35   |
| KNN Regressor               | 3692.40   | 5090.79   | 51390020.71   | 94128237.52   |

![alt text](image.png)

**Random Forest Regressor** menunjukkan performa terbaik pada data uji dengan Test MAE sebesar 1964.85.
---

## Referensi

Altman, N. S. (1992). An introduction to kernel and nearest-neighbor nonparametric regression. The American Statistician, 46(3), 175–185. https://doi.org/10.1080/00031305.1992.10475879

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Koonin, L. M., Hoots, B., Tsang, C. A., Leroy, Z., Farris, K., Jolly, T., ... & Harris, N. S. (2020). Trends in ancillary services for adults with chronic conditions during the COVID-19 pandemic — U.S. Medical Expenditure Panel Survey, 2019–2020. Morbidity and Mortality Weekly Report, 69(45), 1671. https://doi.org/10.15585/mmwr.mm6945a3

Lightwood, J., & Glantz, S. A. (2016). Smoking behavior and healthcare expenditure in the United States, 1992-2009. PloS One, 11(5), e0155100. https://doi.org/10.1371/journal.pone.0155100

Mirichoi0218. (2018). Medical cost personal datasets. Kaggle. Diperoleh dari https://www.kaggle.com/datasets/mirichoi0218/insurance

Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). Introduction to linear regression analysis (6th ed.). John Wiley & Sons. https://www.wiley.com/en-us/Introduction%2Bto%2BLinear%2BRegression%2BAnalysis%2C%2B6th%2BEdition-p-9781119578727

Siegel, E. (2016). Predictive analytics: The power to predict who will click, buy, lie, or die. John Wiley & Sons. https://www.wiley.com/en-us/Predictive%2BAnalytics%3A%2BThe%2BPower%2Bto%2BPredict%2BWho%2BWill%2BClick%2C%2BBuy%2C%2BLie%2C%2Bor%2BDie%2C%2BRevised%2Band%2BUpdated%2BEdition-p-9781119145677

Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. Statistics and Computing, 14(3), 199–222. https://doi.org/10.1023/B:STCO.0000035301.49549.88

World Health Organization. (n.d.). Health financing. WHO. Diperoleh dari https://www.who.int/health-topics/health-financing
