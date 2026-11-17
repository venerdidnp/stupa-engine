/* Module 3 -- real-time textured temple in Three.js.
 *
 * Camera, model matrix, light and colour management are set to MATCH
 * raster3d.py / raytrace.py so `compare.py` is meaningful.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const W = 512, H = 512;
if (new URLSearchParams(location.search).has("auto")) document.body.classList.add("auto");
const canvas = document.getElementById("c");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;      // match srgb.py output encode

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// --- camera: exactly raster3d.render() defaults ---
const camera = new THREE.PerspectiveCamera(45, W / H, 1.0, 14.0);
camera.position.set(2.6, 2.0, 3.4);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.7, 0);
controls.update();

// --- lights: ambient 0.15 + one directional along (0.4, 0.9, 0.6) ---
// Three.js r155+ uses physically-correct light units: Lambert is albedo/PI, so
// intensities must be multiplied by PI to equal raster3d's  albedo * (0.15 + 0.85 N.L).
scene.add(new THREE.AmbientLight(0xffffff, 0.15 * Math.PI));
const sun = new THREE.DirectionalLight(0xffffff, 0.85 * Math.PI);
sun.position.set(0.4, 0.9, 0.6);
scene.add(sun);

// --- parse a binary P6 PPM into a DataTexture (Three.js has no PPM loader) ---
async function loadPPMTexture(url) {
  const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
  let p = 0;
  const tok = () => {
    while (/\s/.test(String.fromCharCode(buf[p]))) p++;
    let s = "";
    while (!/\s/.test(String.fromCharCode(buf[p]))) s += String.fromCharCode(buf[p++]);
    return s;
  };
  if (tok() !== "P6") throw new Error("not a P6 PPM");
  const w = +tok(), h = +tok();
  tok(); p++;                                  // maxval + one whitespace
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = buf[p + i * 3];
    rgba[i * 4 + 1] = buf[p + i * 3 + 1];
    rgba[i * 4 + 2] = buf[p + i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = true;                            // OBJ vt origin is bottom-left
  // match raster3d: bilinear on level 0 only, wrap like np.mod
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.anisotropy = 1;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

// --- helper to resolve asset whether served from root, web/, or GitHub Pages ---
async function resolveAsset(fileName) {
  const candidates = [
    "data/" + fileName,
    "./data/" + fileName,
    "../data/" + fileName,
  ];
  for (const c of candidates) {
    try {
      const res = await fetch(c, { method: "HEAD" });
      if (res.ok) return c;
    } catch {}
  }
  return "../data/" + fileName;
}

let mesh;
let currentTexture = null;

async function loadModel(modelFile) {
  if (mesh) {
    scene.remove(mesh);
    mesh = null;
  }
  try {
    const [modelPath, texPath] = await Promise.all([
      resolveAsset(modelFile),
      resolveAsset("relief.ppm"),
    ]);
    if (!currentTexture) {
      currentTexture = await loadPPMTexture(texPath);
    }
    const obj = await new OBJLoader().loadAsync(modelPath);
    const mat = new THREE.MeshLambertMaterial({ map: currentTexture, side: THREE.FrontSide });
    obj.traverse((c) => { if (c.isMesh) c.material = mat; });
    obj.scale.set(1.4, 1.4, 1.4);
    obj.rotation.y = THREE.MathUtils.degToRad(35);   // rot_y(35) in transforms.py
    mesh = obj;
    scene.add(obj);
  } catch (e) {
    document.getElementById("wrap").insertAdjacentHTML("beforeend",
      `<p style="color:#f66">Could not load model/texture: ${e}. Serve the repo
       root (<code>npx serve .</code>) or web folder and open <code>/web/</code>.</p>`);
  }
}

// Initial load
loadModel("pusaka_model.obj");

const modelSelect = document.getElementById("model-select");
if (modelSelect) {
  modelSelect.addEventListener("change", (e) => {
    loadModel(e.target.value);
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

document.getElementById("wire").addEventListener("click", () => {
  mesh?.traverse((c) => { if (c.isMesh) c.material.wireframe = !c.material.wireframe; });
});

document.getElementById("export").addEventListener("click", () => {
  renderer.render(scene, camera);
  const gl = renderer.getContext();
  const px = new Uint8Array(W * H * 4);
  gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const body = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const s = ((H - 1 - y) * W + x) * 4, d = (y * W + x) * 3;
      body[d] = px[s]; body[d + 1] = px[s + 1]; body[d + 2] = px[s + 2];
    }
  const header = new TextEncoder().encode(`P6\n${W} ${H}\n255\n`);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([header, body]));
  a.download = "temple_three.ppm";
  a.click();
});
