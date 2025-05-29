# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut, sebuah institusi pendidikan perguruan tinggi yang berdiri sejak tahun 2000, menghadapi masalah serius terkait tingginya angka dropout siswa. Meskipun telah mencetak banyak lulusan berreputasi baik, jumlah siswa yang tidak menyelesaikan pendidikan mereka menjadi perhatian utama. Institusi ini berkeinginan untuk mendeteksi siswa yang berpotensi dropout sedini mungkin agar dapat diberikan bimbingan khusus dan intervensi yang tepat waktu.

### Permasalahan Bisnis
- Tingginya dropout yang mengakibatkan kerugian finansial, berdampak negatif pada reputasi institusi, dan menurunkan tingkat kelulusan.
- Kesulitan dalam mengidentifikasi siswa yang berisiko dropout secara dini, menyebabkan intervensi seringkali terlambat atau kurang tepat sasaran.
- Kurangnya pemahaman mendalam mengenai faktor-faktor penyebab utama siswa memutuskan untuk dropout di institusi tersebut.

### Cakupan Proyek
Proyek ini akan mencakup pengembangan solusi data science yang komprehensif untuk mengatasi masalah dropout di Jaya Jaya Institut. Cakupan proyek meliputi:

1. Exploratory Data Analysis (EDA): Memahami karakteristik data siswa dan mengidentifikasi faktor-faktor yang berkorelasi dengan dropout.

2. Pengembangan Model Machine Learning: Membangun dan melatih model prediktif untuk mengidentifikasi siswa yang berpotensi dropout.

3. Pembuatan Business Dashboard: Merancang dashboard interaktif untuk memvisualisasikan data siswa dan memonitor performa mereka, serta memberikan insight yang mudah dipahami bagi manajemen.

4. Pengembangan Prototipe system Machine Learning: Membuat aplikasi berbasis Streamlit sebagai prototipe yang siap digunakan oleh user untuk memprediksi potensi dropout siswa baru atau yang sedang berjalan.

5. Deployment Solusi: Melakukan deployment prototipe system machine learning ke cloud environment (Streamlit Community Cloud) agar dapat diakses secara remote.

6. Penyusunan Rekomendasi: Memberikan rekomendasi action items yang konkret berdasarkan temuan proyek untuk membantu Jaya Jaya Institut mengurangi angka dropout.

### Persiapan

#### Tentang Data

Data yang akan Anda gunakan dalam proyek ini adalah kumpulan data "students' performance" yang disediakan oleh Dicoding Academy, bertujuan untuk membantu Jaya Jaya Institut mengatasi masalah dropout siswa. Dataset ini kemungkinan besar berisi berbagai fitur terkait siswa, seperti informasi demografi (gender, ras/etnis), latar belakang pendidikan orang tua, partisipasi dalam program tertentu (misalnya, kursus persiapan), dan yang terpenting, nilai atau skor akademis siswa dalam mata pelajaran seperti matematika, membaca, dan menulis. Melalui analisis data ini, Anda akan mencari pola dan faktor-faktor yang berkontribusi terhadap status dropout siswa, kemudian memanfaatkan informasi tersebut untuk membangun model prediksi.

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

Format data: CSV, Delimiter: ';'

#### Persiapan Lingkungan Kerja

Berikut adalah persiapan yang harus dilakukan:

1.  **Siapkan Virtual Environment**
    Buka terminal atau command prompt, lalu buat dan aktifkan *virtual environment*. Ini membantu mengelola dependensi proyek Anda:
    ```bash
    python -m venv venv
    ```
    *Aktifkan virtual environment*:
    * **Windows (Powershell):**
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
    * **Windows (Terminal):**
        ```bash
        .\venv\Scripts\activate.bat
        ```
    *Untuk menonaktifkan virtual environment*:
    * **Windows (Terminal):**
        ```bash
        .\venv\Scripts\deactivate.bat
        ```

2.  **Instal Dependensi Proyek**
    Setelah *virtual environment* aktif, instal semua library Python yang dibutuhkan. Jalankan perintah ini dari terminal Anda:
    ```bash
    pip install -r requirements.txt
    ```

## Business Dashboard

Business dashboard telah dikembangkan menggunakan Looker Studio dan dirancang untuk memberikan pemahaman yang mendalam mengenai performa siswa Jaya Jaya Institut serta memonitor potensi dropout. Dashboard ini terdiri dari empat bagian utama yang menyajikan berbagai visualisasi data kunci. Bagian Summary Overview memberikan ringkasan cepat tentang status mahasiswa saat ini, termasuk jumlah total mahasiswa yang telah Lulus, Dropout, dan Aktif Terdaftar. Selanjutnya, bagian Student Demographics berfokus pada karakteristik demografi, menampilkan distribusi umur mahasiswa yang mayoritas didominasi oleh mahasiswa muda, serta distribusi status pernikahan di mana sebagian besar mahasiswa berstatus 'Single'. Dalam bagian Financial Health, dashboard menunjukkan bahwa mahasiswa penerima beasiswa cenderung memiliki tingkat dropout yang lebih rendah. Menariknya, meskipun masalah finansial dapat menjadi pemicu dropout, persentase mahasiswa dropout karena alasan finansial ternyata relatif kecil dibandingkan dengan mereka yang dropout namun memiliki kondisi finansial yang baik, mengindikasikan bahwa keuangan mungkin bukan faktor utama dropout secara keseluruhan. Terakhir, pada bagian Student's Academic Performance, dashboard secara jelas memperlihatkan bahwa mahasiswa yang pada akhirnya dropout cenderung memiliki nilai akademik yang rendah di semester 1 dan semester 2, dan juga tidak menyelesaikan jumlah kursus yang cukup banyak. Secara keseluruhan, dashboard ini dirancang untuk memberikan insight yang dapat ditindaklanjuti untuk mengidentifikasi dan mendukung siswa berisiko tinggi.

Link Dashboard: `https://lookerstudio.google.com/reporting/3a712e4d-c4da-4bda-ba22-0a55fd932e63`

## Menjalankan Sistem Machine Learning

Ikuti langkah-langkah ini untuk menjalankan prototipe system machine learning (pastikan anda sudah membuat virtual environment dan menginstal dependensi yang ada pada file requirements.txt):

1.  **Jalankan Aplikasi Streamlit**
    Pastikan Anda berada di **direktori proyek** (tempat file `app.py` dan folder `model` berada), lalu jalankan aplikasi Streamlit dengan perintah:
    ```bash
    streamlit run app.py
    ```

2.  **Akses Prototipe**
    Streamlit akan otomatis membuka tab baru di *browser web* Anda. Prototipe system *machine learning* akan langsung tampil dan siap digunakan.

    * **Akses Lokal:** Prototipe berjalan di komputer Anda. Anda bisa mengaksesnya melalui URL yang ditampilkan di terminal, biasanya `http://localhost:8501`.

    * **Akses Publik (Online):** Anda juga dapat mengakses prototipe ini melalui tautan publik: `https://randommeower-dropout-detector-app-lqd5r4.streamlit.app/`

## Conclusion
Proyek ini berhasil mengidentifikasi bahwa **performa akademik yang buruk pada semester-semester awal (semester 1 dan 2) merupakan indikator kritis dan akar permasalahan utama yang mengarah pada risiko *dropout* mahasiswa** di Jaya Jaya Institut. Model klasifikasi XGBoost yang dikembangkan dan mendapatkan akurasi sebesar (0.7401), precision sebesar (0.7247), recall sebesar (0.7401), dan f1-score sebesar (0.7306) model ini sudah mampu mendeteksi tanda-tanda peringatan dini ini.

Analisis menunjukkan bahwa **fitur-fitur hasil rekayasa (engineered features)**, seperti **`Overall_Approval_Rate`** (rasio kumulatif SKS lulus terhadap SKS yang diambil, yang dihitung dari data akademik asli per semester), bersama dengan fitur turunan lainnya seperti **`Approval_Rate_1st_Sem`** dan **`Approval_Rate_2nd_Sem`**, menjadi prediktor paling signifikan. Hal ini, didukung oleh data asli jumlah SKS yang disetujui (seperti `Curricular_units_1st_sem_approved` dan `Curricular_units_2nd_sem_approved`) dan perolehan nilai (seperti `Curricular_units_1st_sem_grade` dan `Curricular_units_2nd_sem_grade`) pada periode awal, menggarisbawahi vitalnya fondasi akademik yang kuat. Di samping faktor akademik, kondisi finansial yang tercermin dari kelancaran pembayaran UKT (fitur asli `Tuition_fees_up_to_date`) juga tetap menjadi faktor penting. Lebih lanjut, terindikasi bahwa mahasiswa yang mendapatkan beasiswa (fitur asli `Scholarship_holder`) cenderung memiliki risiko *dropout* yang lebih rendah, menyoroti dampak positif dari dukungan finansial terhadap kelangsungan studi. 

Dengan pemahaman mendalam ini, Jaya Jaya Institut dapat memfokuskan sumber dayanya untuk intervensi super dini, membantu mahasiswa mengatasi kesulitan sejak awal perkuliahan. Prototipe aplikasi Streamlit dan *dashboard* yang direncanakan akan menjadi alat bantu vital dalam upaya proaktif ini, demi menekan angka *dropout* dan meningkatkan tingkat kelulusan.


### Rekomendasi Action Items
Berikut adalah rekomendasi tindakan ringkas untuk Jaya Jaya Institut:

1.  **Tingkatkan Seleksi & Program Pra-Kuliah:** Tingkatkan kriteria seleksi dan adakan program matrikulasi atau persiapan pra-kuliah untuk calon mahasiswa guna kesiapan yang lebih baik.

2.  **Aktifkan Deteksi Dini Performa Semester 1 & 2:** Gunakan model prediksi untuk identifikasi cepat mahasiswa dengan performa akademik di bawah standar pada semester awal, khususnya pada tingkat kelulusan SKS dan perolehan nilai.

3.  **Sediakan Dukungan Akademik & Psikologis Dini:** Tawarkan bimbingan akademik intensif, mentoring, serta akses mudah ke layanan konseling dan psikolog bagi mahasiswa yang teridentifikasi berisiko untuk dropout sejak semester awal.

4.  **Optimalkan Skema Bantuan Finansial & Perluas Akses Beasiswa:** Secara proaktif tawarkan solusi bagi mahasiswa yang mengalami kesulitan finansial (berdasarkan status pembayaran UKT) dan perluas serta sosialisasikan program beasiswa.

5.  **Lakukan Review dan Iterasi Program Intervensi Berkala:** Evaluasi dan sesuaikan secara periodik efektivitas model prediksi serta semua program intervensi (pra-kuliah, akademik, finansial, psikologis) untuk optimalisasi berkelanjutan.
