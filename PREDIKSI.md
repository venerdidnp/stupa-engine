# LEMBAR PREDIKSI MAHASISWA (Lab 3 no. 3)

> **PENTING**: Lembar ini harus diisi oleh mahasiswa **SEBELUM** menjalankan `python bench_frametime.py`. Jangan mengubah angka atau isi setelah benchmark dijalankan.

---

### Informasi Mahasiswa & Waktu Pengisian
- **Nama Mahasiswa**            : Venerdi Dinarsa
- **NIM**                       : [ISI NIM ANDA]
- **Tanggal dan Jam Ditulis**   : 2026-10-18 06:40 WIB

---

### 1. Prediksi Rasio Frame Time (raster3d : raytrace)
- **Prediksi Rasio 256x256**    : 1 : 5 (raster3d diperkirakan ~5x lebih cepat dibandingkan raytrace)
- **Prediksi Rasio 512x512**    : 1 : 7 (raster3d diperkirakan ~7x lebih cepat dibandingkan raytrace)
- **Alasan / Dasar Tebakan**    : 
  - Pada algoritma rasterisasi (`raster3d`), kompleksitas waktu didorong oleh jumlah segitiga ($O(T)$) dan area piksel yang dicakup (*screen-space bounding box fragments*), didukung optimasi *back-face culling* yang memotong ~50% segitiga sebelum pengujian fragment/kedalaman dilakukan.
  - Sebaliknya, algoritma ray tracer (`raytrace`) pada repositori ini bekerja secara naif tanpa struktur data percepatan spasial (seperti BVH atau k-d tree), sehingga setiap piksel menembakkan sinar primer yang harus diuji perpotongannya (Möller–Trumbore) terhadap seluruh 312 segitiga ($O(W \times H \times T)$).
  - Ketika resolusi dinaikkan dari 256x256 ke 512x512 (peningkatan jumlah sinar 4 kali lipat dari 65.536 ke 262.144 sinar, setara ~81,8 juta tes perpotongan), beban komputasi ray tracer melonjak secara kuadratik terhadap dimensi layar. Sementara pada rasterizer, *overhead* transformasi matriks per-vertex dan *setup* segitiga bersifat tetap, sehingga rasio keunggulan kecepatan raster3d diprediksi semakin besar.

---

### 2. Kualitas Shadow (Bayangan)
- **Renderer Pilihan**          : raytrace
- **Alasan**                    : 
  - Ray tracing mengevaluasi bayangan (*shadow rays*) secara alami dan berbasis fisika: cukup menembakkan sinar sekunder dari titik impak (*hit point*) langsung menuju posisi sumber cahaya untuk menguji oklusi visibilitas. Teknik ini menghasilkan bayangan tajam (*hard shadow*) maupun bayangan lembut (*soft shadow*) yang akurat tanpa artefak resolusi.
  - Sebaliknya, rasterisasi memerlukan algoritma aproksimasi tambahan seperti *Shadow Mapping* yang rentan terhadap artefak diskritisasi kedalaman (misalnya *shadow acne*, *peter panning*, dan *perspective aliasing*) serta dibatasi oleh resolusi buffer kedalaman (*depth map*).

---

### 3. Kualitas Reflection (Pantulan)
- **Renderer Pilihan**          : raytrace
- **Alasan**                    : 
  - Ray tracing mampu mensimulasikan pantulan cermin (*specular reflection*) dan pencahayaan global rekursif (*Whitted-style ray tracing*) secara presisi. Vektor pantulan dihitung langsung ($r = d - 2(d \cdot n)n$) dan dapat melacak geometri apa pun di seluruh ruang 3D dunia, termasuk objek yang berada di belakang atau di luar sudut pandang kamera.
  - Pada rasterisasi, penggambaran bersifat lokal per-primitif di layar. Pantulan membutuhkan teknik aproksimasi seperti *Cube Environment Mapping* (statis atau memerlukan multi-pass mahal) atau *Screen Space Reflections (SSR)* yang hanya bisa memantulkan objek yang sedang tampak di layar (menghasilkan *cutoff* atau visual yang menghilang ketika geometri pemantul keluar dari frustum).

---

### 4. Pilihan untuk Viewer Real-Time
- **Renderer Pilihan**          : raster3d
- **Alasan**                    : 
  - Penampil real-time interaktif (seperti 3D viewer web atau game) memprioritaskan *latency* yang sangat rendah serta *frame rate* tinggi dan stabil (minimal 30–60 FPS).
  - Rasterisasi dirancang khusus untuk *throughput* tinggi pada kartu grafis/pipeline paralel dengan efisiensi tinggi melalui *frustum clipping*, *back-face culling*, dan *early depth testing*. Ray tracing naif berbasis CPU membutuhkan waktu komputasi yang terlalu besar per frame untuk interaksi pengguna secara langsung.

---

### 5. Pilihan untuk Beauty Shot Marketing
- **Renderer Pilihan**          : raytrace
- **Alasan**                    : 
  - *Beauty shot* pemasaran berfokus penuh pada fidelitas visual tertinggi, realisme fotorealistik, keakuratan kontak bayangan, pantulan permukaan material, serta pencahayaan global (*global illumination*).
  - Dalam skenario *offline rendering* untuk materi komersial/poster diam, waktu komputasi (beberapa detik hingga menit per *frame*) bukanlah masalah dibandingkan dengan kualitas hasil akhir yang dihasilkan. Oleh karena itu, ray tracing adalah pilihan superior untuk menghasilkan citra dengan kualitas sinematik maksimal.

