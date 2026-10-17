Dokumen ini hanya bahan. Bagian bertanda [TULIS SENDIRI] harus diisi mahasiswa dengan kata-katanya sendiri.

# BAHAN TUGAS ANALISIS (LAB 1, 2, DAN 3)

---

## A1. Telusuran Vertex & Worked Example

### 1. Telusuran Vertex Model (`lab1_trace.py --vertex 0`)
Sumber bukti: [KUNCI_LAB1.txt](file:///c:/Users/venerdi/Documents/stupa-engine/KUNCI_LAB1.txt)

- **Parameter Kamera**:
  - Eye: `[11.0, 8.0, 11.0]`
  - Target: `[0.0, 4.6, 0.0]`
  - Up: `[0.0, 1.0, 0.0]`
  - FOV: `50.0°`, Near: `1.0`, Far: `80.0`
  - Viewport: `800 x 600`
- **Matriks M (Model Matrix)**:
  ```
  [  1.000000   0.000000   0.000000   0.000000 ]
  [  0.000000   1.000000   0.000000   0.000000 ]
  [  0.000000   0.000000   1.000000   0.000000 ]
  [  0.000000   0.000000   0.000000   1.000000 ]
  ```
- **Matriks V (View Matrix)**:
  ```
  [  0.707107   0.000000  -0.707107  -0.000000 ]
  [ -0.150981   0.976939  -0.150981  -4.493918 ]
  [  0.690800   0.213520   0.690800 -16.905759 ]
  [  0.000000   0.000000   0.000000   1.000000 ]
  ```
- **Matriks P (Projection Matrix)**:
  ```
  [  1.608380   0.000000   0.000000   0.000000 ]
  [  0.000000   2.144507   0.000000   0.000000 ]
  [  0.000000   0.000000  -1.025316  -2.025316 ]
  [  0.000000   0.000000  -1.000000   0.000000 ]
  ```
- **Baca Kolom M**:
  - Origin model `(0,0,0,1)` $\rightarrow$ `[0.000, 0.000, 0.000, 1.000]`
  - Arah +x model `(1,0,0,0)` $\rightarrow$ `[1.000, 0.000, 0.000, 0.000]`
  - Arah +y model `(0,1,0,0)` $\rightarrow$ `[0.000, 1.000, 0.000, 0.000]`
  - Arah +z model `(0,0,1,0)` $\rightarrow$ `[0.000, 0.000, 1.000, 0.000]`
- **Baca Kolom V**:
  - Origin world `(0,0,0,1)` $\rightarrow$ `[-0.000, -4.494, -16.906, 1.000]`
  - Arah +x world `(1,0,0,0)` $\rightarrow$ `[0.707, -0.151, 0.691, 0.000]`
  - Arah +y world `(0,1,0,0)` $\rightarrow$ `[0.000, 0.977, 0.214, 0.000]`
  - Arah +z world `(0,0,1,0)` $\rightarrow$ `[-0.707, -0.151, 0.691, 0.000]`
- **Telusuran Koordinat Vertex 0**:
  - 1. Object Space : `[-3.000, 0.000, -3.000, 1.000]`
  - 2. World Space  : `[-3.000, 0.000, -3.000, 1.000]`
  - 3. View Space   : `[-0.000, -3.588, -21.051, 1.000]`
  - 4. Clip Space   : `[-0.000, -7.695, 19.558, 21.051]`
  - 5. NDC Space    : `[-0.000, -0.366, 0.929]`
  - 6. Screen Space : pixel x = `400.000`, y = `409.658`, depth01 = `0.965`

### 2. Angka Worked Example (`worked_example.py`)
- Titik uji input: `p = (2.0, 3.0, 4.0, 1.0)`
- Transformasi Model: `M = rot_y(90) @ translate(1, 0, -1)`
- World Space  : `(3.000, 3.000, -3.000, 1.000)`
- Eye / View   : `(3.000, 3.000, -11.000, 1.000)`
- Clip Space   : `(5.196, 5.196, 9.408, 11.000)`
- NDC Space    : `(0.472, 0.472, 0.855)`
- Screen Space : pixel `(589.000, 158.000)`, depth01 = `0.928`

---

## A2. Angka Pengujian E1–E4 & Perbandingan Renderer

### 1. Rangkuman Angka Tahap E1–E4 (`tests.py`)
- **E1 (Transforms & Projections)**:
  - `(2,3,4)` translate `(1,0,-1)` lalu `rot_y(90)` $\rightarrow$ `(3.0, 3.0, -3.0)`
  - Perspective divide focal distance $d=2$ $\rightarrow$ `(2.0, 2.0)`
  - `(2,3,4)` `rot_y(90)` lalu translate `(1,0,-1)` $\rightarrow$ `(5.0, 3.0, -3.0)`
  - Kamera geser $+1$ di X vs Model geser $-1$ di X $\rightarrow$ keduanya menghasilkan eye coordinate `(4.0, 0.0, 0.0)`
  - Parameter depth perspective ($n=1, f=9$): $A = -1.25$, $B = -2.25$, pemetaan depth $z \in \{-1, -9, -3\} \rightarrow z_{ndc} \in \{-1.0, +1.0, 0.5\}$
- **E2 (Aset & I/O)**:
  - Model `data/temple.obj`: 208 vertex, 624 UV planar (24 per box), 312 segitiga
  - Bounding box model: $x \in [-3.0, 3.0]$, $y \in [0.0, 10.6]$, $z \in [-3.0, 3.0]$
  - Tekstur `data/relief.ppm`: resolusi 256 x 256 RGB, procedural sandstone
- **E3 (Rasterizer)**:
  - Interpolasi barycentric pada bobot `(0.25, 0.25, 0.5)` dengan nilai depth `0.20/0.50/0.90` $\rightarrow$ `0.625`
  - Z-buffer overlap depth $0.3$ vs $0.7$: fragmen near (`-0.4`) selalu menang atas far (`+0.4`)
  - Near clipping Sutherland-Hodgman: memotong segitiga melintasi bidang dekat secara eksak dengan $z + w \ge 0$
  - Piramida Mipmap: 4 level (256x256, 128x128, 64x64, 32x32)
- **E4 (Perspective UV)**:
  - Tepi $w_L=1, w_R=9$: midpoint naif = `0.5`, midpoint perspective-correct = `0.1`

### 2. Perbandingan Rasterizer vs Ray Tracer (`compare.py out/temple_raster.ppm out/temple_ray.ppm`)
- **Resolusi**: 800 x 600
- **PSNR**: `inf dB` (kedua citra identik byte-for-byte)
- **SSIM**: `1.000000`
- **Mean |diff|**: `0.000000 / 255`
- **Piksel Berbeda**: `0` dari 480.000 piksel (0.00%)
- **Maksimum Selisih Depth**: $3.33 \times 10^{-16}$ (presisi float64)

### 3. Perbandingan NumPy Raster vs WebGL Three.js (`webshot.py`)
- **Resolusi**: 512 x 512
- **PSNR**: `39.84 dB`
- **SSIM**: `0.992226`
- **Mean |diff|**: `1.4646 / 255`
- **Sebaran Perbedaan**: Semua perbedaan berada tepat di tepi siluet (dalam radius 2 piksel dari tepi warna akibat antialiasing MSAA pada canvas WebGL browser; bagian interior model identik).

### 4. Ketersediaan File Referensi
- **Catatan**: File `ref/temple_ref.ppm` **TIDAK TERSEDIA** di dalam repositori.

---

## A3. Showcase Tiga Bug Rendering (`broken_scene.py`)

| Kasus Bug | Perintah Eksekusi | Metrik Sebelum Perbaikan | Metrik Sesudah Perbaikan | File Bukti Gambar | DIAGNOSIS PRA-PERBAIKAN | PITFALL §12 YANG TERKAIT |
| --- | --- | --- | --- | --- | --- | --- |
| **Inside-Out (Winding Terbalik)** | `python broken_scene.py --bug inside-out` | Culled: 154 segitiga<br>Probe depth: 0.957580<br>PSNR: 11.23 dB<br>SSIM: 0.742308 | Culled: 158 segitiga<br>Probe depth: 0.945443<br>PSNR: inf dB<br>SSIM: 1.000000<br>(0 piksel beda) | `out/broken_inside-out.png`<br>`out/broken_inside-out_diff.png` | [TULIS SENDIRI] | [TULIS SENDIRI] |
| **Z-Fight (Near Plane 1e-3)** | `python broken_scene.py --bug zfight` | Depth codes 8-bit: 1 kode (254)<br>Depth codes 16-bit: 3 dari 65536 kode<br>Flicker (dither 0.5 LSB): 398 piksel<br>PSNR: 29.80 dB<br>SSIM: 0.993072 | Depth codes 8-bit: 4 kode (241..244)<br>Flicker pada near=1.0: 0 piksel<br>PSNR: inf dB<br>SSIM: 1.000000<br>(0 piksel beda) | `out/broken_zfight.png`<br>`out/broken_zfight_diff.png` | [TULIS SENDIRI] | [TULIS SENDIRI] |
| **Stretched (Matriks V @ P)** | `python broken_scene.py --bug stretched` | Covered pixels: 19.200 (100% frame)<br>PSNR: 8.78 dB<br>SSIM: 0.158564 | Covered pixels: 3.292 piksel<br>PSNR: inf dB<br>SSIM: 1.000000<br>(0 piksel beda) | `out/broken_stretched.png`<br>`out/broken_stretched_diff.png` | [TULIS SENDIRI] | [TULIS SENDIRI] |

---

## A4. Benchmark Frame Time: Rasterizer vs Ray Tracer (`bench_frametime.py`)

> *Catatan: Tabel ini diisi setelah mahasiswa mengisi `PREDIKSI.md` dan mengeksekusi `python bench_frametime.py`.*

| Resolusi | Renderer | Waktu Median (s) | Waktu Min (s) | Waktu Max (s) | Rasio Terukur (Ray/Raster) | PREDIKSI vs TERUKUR |
| --- | --- | --- | --- | --- | --- | --- |
| **256 x 256** | raster3d | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [DIISI MAHASISWA] | 1.00x | [TULIS SENDIRI] |
| | raytrace | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [TULIS SENDIRI] |
| **512 x 512** | raster3d | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [DIISI MAHASISWA] | 1.00x | [TULIS SENDIRI] |
| | raytrace | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [DIISI MAHASISWA] | [TULIS SENDIRI] |

### Kondisi Pilihan Berbalik (Kapan Ray Tracer Lebih Unggul / Sebaliknya)
- **Kondisi 1**: [TULIS SENDIRI]
- **Kondisi 2**: [TULIS SENDIRI]

---

## A5. Bukti Perspective-Correct UV & Mipmapping

### 1. Data Perspective-Correct UV (`e4_demo.py`)
- **Perhitungan Titik Tengah Layar ($t = 0.5$, $w_L=1, w_R=9$)**:
  - Interpolasi naif: $u = 0.500000$
  - Interpolasi perspective-correct: $u = 0.100000$
  - Selisih nilai $u$: $0.400000$ (kesalahan $40\%$ dari rentang UV)
- **Render Lantai Quad Kotak-Kotak (600 x 450)**:
  - PSNR: `7.99 dB`
  - SSIM: `0.290695`
  - Mean |diff|: `51.0566 / 255`
  - Max |diff|: `210.0000 / 255`
  - Bukti Citra:
    - [out/e4_naive.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/e4_naive.png)
    - [out/e4_correct.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/e4_correct.png)
    - [out/e4_diff.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/e4_diff.png)

### 2. Data Mipmapping & Anti-Aliasing (`mip_demo.py`)
- **Kamera Jarak Jauh (Minifikasi Tekstur)**:
  - Eye frame 1: `(33.00, 14.80, 33.00)`
  - Eye frame 2 (geser 0.01): `(33.01, 14.80, 33.00)`
- **Kelap-Kelip (Flicker Antar-Frame Tergeser)**:
  - Mean |diff| Tanpa Mipmap: `0.006088 / 255`
  - Mean |diff| Dengan Mipmap: `0.001575 / 255`
  - Penurunan Kelap-Kelip: **`74.13%`**
- **Energi Frekuensi Tinggi (Mean |Laplacian|)**:
  - Energi Tanpa Mipmap: `0.622248`
  - Energi Dengan Mipmap: `0.478458`
  - Penurunan Frekuensi Tinggi: **`23.11%`**
- **Perbandingan Citra (Dengan vs Tanpa Mipmap)**:
  - PSNR: `51.65 dB`
  - SSIM: `0.998384`
  - Mean |diff|: `0.0691 / 255`
  - Bukti Citra:
    - [out/mip_without.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/mip_without.png)
    - [out/mip_with.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/mip_with.png)
    - [out/mip_diff_modes.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/mip_diff_modes.png)
    - [out/mip_diff_flicker_without.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/mip_diff_flicker_without.png)
    - [out/mip_diff_flicker_with.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/mip_diff_flicker_with.png)

### 3. Data Pitfall Back-face Culling & Normal Matrix (`pitfall_demo.py`)
- **Back-Face Culling (800 x 600)**:
  - Segitiga disubmit: 312
  - Segitiga di-cull: 158 (cull=True) vs 0 (cull=False)
  - Fragment lolos z-test: 139.546 (cull=True) vs 185.776 (cull=False) (penurunan 24.88%)
  - Waktu render median: 0.2813 s vs 0.3821 s (penghematan waktu 26.40%)
  - Selisih citra: 0 piksel beda (PSNR = inf dB, SSIM = 1.000000)
- **Normal Matrix Inverse-Transpose**:
  - Model scale(1,3,1) axis-aligned: PSNR = `inf dB`, SSIM = `1.000000`, mean |diff| = `0.0000` (karena normal kotak sejajar sumbu kartesius tidak berubah arah saat diskalakan searah sumbu).
  - Model scale(1,3,1) @ rot_z(30) (permukaan miring): PSNR = `31.10 dB`, SSIM = `0.986393`, mean |diff| = `2.1704 / 255`, max |diff| = `27.0000 / 255`.
  - Bukti Citra:
    - [out/normal_wrong.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/normal_wrong.png)
    - [out/normal_correct.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/normal_correct.png)
    - [out/normal_diff.png](file:///c:/Users/venerdi/Documents/stupa-engine/out/normal_diff.png)

---

## A6. Transkrip Clean Clone & Status GitHub Pages

### 1. Transkrip Clean Clone (`verify_clean.ps1`)
- **Tanggal Eksekusi**: 2026-10-17
- **Sistem Operasi**: Windows 11 Home Single Language (10.0.26200)
- **Toolchain**: Python 3.11.9, NumPy 2.3.5, Pillow 12.2.0, Node.js v24.13.0
- **Tahapan & Hasil**:
  1. `git clone` ke direktori temporary: OK
  2. Pembuatan virtualenv terisolasi `.venv`: OK
  3. `pip install -r requirements.txt`: OK (numpy 2.3.5, pillow 12.2.0 terpasang)
  4. `python make_model.py`: OK (menghasilkan 208 vertex, 624 UV, 312 segitiga)
  5. `python tests.py`: OK (**130 dari 130 checks passed**, exit code 0)
  6. Web server `npx -y serve -l 39821 web`: OK (HTTP Response Status: **200 OK**)
  7. Pembersihan folder temporary: OK
- **Total Waktu Verifikasi**: 52.28 detik

### 2. Status GitHub Pages
- **URL**: `https://venerdidnp.github.io/stupa-engine/`
- **Kode Status HTTP**: **404 Not Found**
- **Analisis Konfigurasi (`.github/workflows/pages.yml`)**:
  - File workflow `.github/workflows/pages.yml` sudah terkonfigurasi dengan benar (menggunakan `actions/upload-pages-artifact@v3` dan `actions/deploy-pages@v4`).
  - **Penyebab HTTP 404**: Fitur GitHub Pages belum diaktifkan pada repository settings di GitHub Web UI. Pengaturan yang diperlukan: masuk ke menu repository **Settings $\rightarrow$ Pages $\rightarrow$ Build and deployment $\rightarrow$ Source**: ubah menjadi **GitHub Actions**. Selain itu, commit lokal saat ini belum dipush ke remote origin (`origin/main`).
