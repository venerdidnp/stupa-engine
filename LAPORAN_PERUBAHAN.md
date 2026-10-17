# LAPORAN PERUBAHAN & KELENGKAPAN LAB 1, 2, DAN 3
**Praktikum Grafika Komputer Modul 3 (stupa-engine)**
**Tanggal**: 2026-10-17

---

## 1. Daftar File Baru & Diubah

### File yang Diubah:
1. `make_model.py`:
   - Memperluas pembuatan model candi dari 15 box (180 segitiga, 120 vertex) menjadi **26 box (312 segitiga, 208 vertex, 624 vt)** dengan menambahkan 4 stupa sudut pada `base_1`, 4 stupa sudut pada `base_2`, dan 3 anak tangga di sisi depan (+z).
   - Mempertahankan winding CCW dari luar dan UV planar per-face yang deterministik.
   - Menyalin output aset secara otomatis ke `data/` dan `web/data/`.
2. `tests.py`:
   - Memperbarui pengecekan jumlah vertex (208), UV (624), segitiga (312), loop pengujian winding CCW seluruh box (312), jumlah segitiga submit (312), culling flip `inside-out` ([158, 154]), dan luas foreground `stretched` (3292).
3. `raster3d.py`:
   - Menambahkan opsi `perspective_correct: bool = True` dan `normal_matrix_mode: str = "correct"` pada `RenderSettings`.
   - Menambahkan dukungan argumen `m_matrix` dan penanganan transformasi normal (direct $M$ vs inverse-transpose $(M^{-1})^T$) pada `render_scene`.
   - Menambahkan opsi interpolasi UV naif (`interp_linear`) vs perspective-correct (`interp_perspective`) pada `rasterize_triangle`.
   - Seluruh nilai default dipertahankan sehingga tidak mengubah perilaku atau output lama.
4. `README.md`:
   - Menambahkan pernyataan jujur bahwa model dibuat secara prosedural oleh `make_model.py` (bukan manual Blender).
   - Menambahkan baris verifikasi clean clone: `Verified: 2026-10-17 | Windows 11 (10.0.26200) | Python 3.11.9 | NumPy 2.3.5 | Node v24.13.0 | clean clone ok`.
   - Menambahkan tabel parameter kamera perbandingan NumPy vs WebGL 512x512.
   - Menambahkan tabel toolchain lingkungan pengujian.

### File Baru:
1. `uv_orientation_check.py`:
   - Merender model candi menggunakan tekstur asimetris (huruf 'F' dan penanda orientasi) buatan NumPy tanpa menimpa `relief.ppm`.
   - Menyimpan `out/asymmetric_marker.ppm`, `out/uv_orientation.ppm`, dan `out/uv_orientation.png`.
2. `lab1_trace.py`:
   - Melakukan telusuran satu vertex (`--vertex INDEX`, default 0) melalui ruang Object, World, View, Clip, NDC, dan Screen.
   - Mencetak matriks M, V, P, dan "baca kolom" untuk origin dan sumbu basis.
   - Menyimpan output ke `KUNCI_LAB1.txt` dengan header *"KUNCI JAWABAN - hitung dulu dengan tangan, baru bandingkan"*.
3. `pitfall_demo.py`:
   - Mengukur efektivitas back-face culling (jumlah fragment z-test, segitiga di-cull, penurunan persen, dan median waktu render 5 iterasi).
   - Mengukur transformasi normal matrix inverse-transpose vs direct $M$ pada scaling non-seragam.
   - Menyimpan `out/normal_wrong.png`, `out/normal_correct.png`, `out/normal_diff.png`, dan laporan angka ke `out/pitfall_report.txt`.
4. `e4_demo.py`:
   - Menghitung langkah-demi-langkah interpolasi UV naif ($u=0.5$) vs perspective-correct ($u=0.1$) pada tepi $w_L=1, w_R=9$.
   - Merender lantai quad kotak-kotak (checkerboard) menjauh dari kamera dengan mode naif vs perspective-correct.
   - Menyimpan `out/e4_naive.png`, `out/e4_correct.png`, dan `out/e4_diff.png`.
5. `mip_demo.py`:
   - Merender candi dari jarak jauh dengan dan tanpa mipmap.
   - Mengukur kelap-kelip (*flicker* mean |diff|) pada pergeseran kamera 0.01 unit dan energi frekuensi tinggi (mean |Laplacian|).
   - Menyimpan citra, citra selisih, dan laporan ke `out/mip_report.txt`.
6. `PREDIKSI.md`:
   - Template kosong bagi mahasiswa untuk menulis prediksi sebelum menjalankan benchmark.
7. `bench_frametime.py`:
   - Skrip benchmark frame time raster3d vs raytrace pada 256x256 dan 512x512 (minimal 5 kali jalan).
   - **TIDAK DIJALANKAN** sesuai instruksi (menunggu pengisian `PREDIKSI.md` oleh mahasiswa).
8. `verify_clean.ps1`:
   - Skrip otomatisasi PowerShell untuk kloning repositori ke folder temporary, pembuatan virtualenv terisolasi, instalasi dependensi, eksekusi `make_model.py`, `tests.py`, pengetesan web server `npx -y serve -l 39821 web` (HTTP 200 OK), dan pembersihan folder temporary.
9. `KUNCI_LAB1.txt`:
   - Hasil telusuran koordinat vertex 0 untuk kunci jawaban mandiri mahasiswa.
10. `ANALISIS_BAHAN.md`:
    - Kumpulan data mentah, tabel, dan tautan bukti pengujian untuk mempermudah pengerjaan tugas analisis mahasiswa tanpa teks interpretasi.

---

## 2. Penyesuaian Tes pada `tests.py`

| Lokasi Baris | Nama / Pengecekan Tes | Nilai Lama | Nilai Baru | Alasan Penyesuaian |
| --- | --- | --- | --- | --- |
| Baris 242 | `temple.obj: vertices = ...` | 120 | 208 | Model diperluas dari 15 box menjadi 26 box ($26 \times 8 = 208$ vertex). |
| Baris 243 | `temple.obj: uvs = ...` | 360 | 624 | 26 box masing-masing memiliki 24 planar UV ($26 \times 24 = 624$). |
| Baris 244 | `temple.obj: triangles = ...` | 180 | 312 | 26 box masing-masing memiliki 12 segitiga ($26 \times 12 = 312 \ge 260$). |
| Baris 272–280 | `temple.obj: all ... triangles wound CCW seen from outside` | 180 (loop 15) | 312 (loop `num_boxes = 26`) | Memverifikasi seluruh 312 segitiga pada 26 box memiliki normal mengarah ke luar box (CCW). |
| Baris 534 | `full render: ... triangles submitted` | 180 | 312 | Jumlah segitiga model yang disubmit ke rasterizer menjadi 312 segitiga. |
| Baris 557 | Print independent back-face count | of 180 | of 312 | Format teks pencocokan independen back-face culling disesuaikan ke total 312 segitiga. |
| Baris 909–911 | `inside-out: culled count flips (... correct -> ... broken)` | [92, 88] | [158, 154] | Pada model 312 segitiga, kamera default meng-cull 158 segitiga pada arah normal benar, dan 154 segitiga saat winding dibalik ($158 + 154 = 312$). |
| Baris 957–959 | `stretched: the broken chain covers the whole frame` | [3258, 19200] | [3292, 19200] | Penambahan geometri tangga depan dan stupa sudut menambah luas piksel foreground yang terproyeksikan pada frame 160x120 menjadi 3292 piksel. |

---

## 3. Hasil Pengujian Terakhir (`python tests.py`)

- **Status**: **132 dari 132 checks passed (100% lulus)**
- **Exit Code**: `0`
- **Raster vs Ray Trace**: `PSNR = inf dB` (identik byte-for-byte), `max |diff| depth = 3.33e-16`.

---

## 4. Daftar Hal yang Tidak Berhasil Dikerjakan / Catatan Kejujuran

1. **Benchmark `bench_frametime.py` Sengaja Tidak Dijalankan**:
   - Sesuai instruksi Bagian F No. 3: agen diwajibkan berhenti dan tidak mengeksekusi benchmark agar mahasiswa mengisi lembar `PREDIKSI.md` terlebih dahulu dengan prediksi dan tanggal/jam sebelum pengukuran nyata dilakukan.
2. **Hard Shadow pada Pipeline Dasar**:
   - Fitur bayangan tajam (*hard shadow*) tidak diimplementasikan pada pipeline rasterizer karena arsitektur bawaan adalah single-pass forward rasterizer tanpa shadow map pass sekunder. Sesuai instruksi, dicatat secara jujur: *"shadow tidak diimplementasi pada pipeline dasar"*.
3. **Ketersediaan File `ref/temple_ref.ppm`**:
   - File referensi lama `ref/temple_ref.ppm` tidak tersedia di dalam repositori stupa-engine.
4. **Status HTTP GitHub Pages**:
   - Permintaan HTTP ke `https://venerdidnp.github.io/stupa-engine/` menghasilkan kode status **404 Not Found**. Berdasarkan pengecekan file `.github/workflows/pages.yml`, alur kerja GitHub Actions sudah benar, namun repositori di GitHub belum diaktifkan fitur Pages-nya melalui menu *Settings > Pages > Source: GitHub Actions*, serta commit lokal belum di-push ke remote origin. Sesuai instruksi, pengaturan repository di remote tidak diubah.
