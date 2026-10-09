import "./styles.css";

import { ImageOrbitController } from "./image-orbit-controller.js";

const byId = (id) => document.getElementById(id);

const elements = {
  shell: byId("app-shell"),
  datasetName: byId("dataset-name"),
  pathCount: byId("path-count"),
  sampleCount: byId("sample-count"),
  modelCount: byId("model-count"),
  imageTab: byId("image-mode-tab"),
  plyTab: byId("ply-mode-tab"),
  imagePanel: byId("image-mode-panel"),
  plyPanel: byId("ply-mode-panel"),
  bootTitle: byId("boot-title"),
  bootDetail: byId("boot-detail"),
  errorBanner: byId("error-banner"),
  errorMessage: byId("error-message"),
};

const imageElements = {
  stage: byId("image-stage"),
  image: byId("orbit-image"),
  heading: byId("orbit-heading"),
  coordinates: byId("orbit-coordinates"),
  pathLabel: byId("active-path-label"),
  frameAngle: byId("frame-angle"),
  slider: byId("frame-slider"),
  matchError: byId("image-match-error"),
  frameId: byId("image-frame-id"),
  framePath: byId("image-frame-path"),
  routeList: byId("route-list"),
};

const plyElements = {
  stage: byId("ply-stage"),
  canvasHost: byId("ply-canvas-host"),
  loading: byId("ply-loading"),
  loadingTitle: byId("ply-loading-title"),
  loadingDetail: byId("ply-loading-detail"),
  loadBar: byId("ply-load-bar"),
  modelScaleInput: byId("model-scale-input"),
  modelScaleValue: byId("model-scale-value"),
  toggleCamerasButton: byId("toggle-cameras-button"),
  focusModelButton: byId("focus-model-button"),
  resetButton: byId("reset-view-button"),
  pathLabel: byId("ply-path-label"),
  matchError: byId("ply-match-error"),
  comparisonImageWrap: document.querySelector(".comparison-image-wrap"),
  comparisonImage: byId("comparison-image"),
  frameId: byId("ply-frame-id"),
  frameAngle: byId("ply-frame-angle"),
  cameraPosition: byId("ply-camera-position"),
};

let manifest = null;
let imageController = null;
let plyViewer = null;
let plyInitialization = null;
let currentMode = "image";

function showError(error) {
  const message = error instanceof Error ? error.message : String(error);
  elements.errorMessage.textContent = message;
  elements.errorBanner.hidden = false;
  console.error(error);
}

function updateTabState(mode) {
  const imageActive = mode === "image";
  elements.imageTab.classList.toggle("is-active", imageActive);
  elements.plyTab.classList.toggle("is-active", !imageActive);
  elements.imageTab.setAttribute("aria-selected", String(imageActive));
  elements.plyTab.setAttribute("aria-selected", String(!imageActive));
  elements.imagePanel.hidden = !imageActive;
  elements.plyPanel.hidden = imageActive;
}

async function setMode(mode) {
  if (!manifest || !["image", "ply"].includes(mode)) return;
  currentMode = mode;
  updateTabState(mode);
  imageController.setActive(mode === "image");

  if (mode === "ply") {
    if (!plyViewer) {
      const { PlyOrbitViewer } = await import("./ply-orbit-viewer.js");
      plyViewer = new PlyOrbitViewer(manifest, plyElements, showError);
    }
    plyViewer.setActive(true);
    if (!plyInitialization) {
      plyInitialization = plyViewer.initialize().catch((error) => {
        plyInitialization = null;
        throw error;
      });
    }
    try {
      await plyInitialization;
    } catch {
      return;
    }
  } else if (plyViewer) {
    plyViewer.setActive(false);
  }
}

async function start() {
  try {
    elements.bootTitle.textContent = "建立相機索引";
    elements.bootDetail.textContent = "正在讀取 12 條影像路徑";
    const response = await fetch("/api/manifest", { cache: "no-cache" });
    if (!response.ok) throw new Error(`資料 API 回傳 ${response.status}`);
    manifest = await response.json();

    if (manifest.stats.path_count !== 12) {
      throw new Error(`需要 12 條路徑，目前找到 ${manifest.stats.path_count} 條`);
    }

    const formatter = new Intl.NumberFormat("zh-Hant");
    elements.datasetName.textContent = `${manifest.dataset_name} / ${manifest.model.name}`;
    elements.pathCount.textContent = formatter.format(manifest.stats.path_count);
    elements.sampleCount.textContent = formatter.format(manifest.stats.total_camera_samples);
    elements.modelCount.textContent = formatter.format(manifest.model.vertex_count);

    elements.bootTitle.textContent = "載入第一個視角";
    elements.bootDetail.textContent = `${manifest.image_size[0]} × ${manifest.image_size[1]} PNG`;
    imageController = new ImageOrbitController(manifest, imageElements);
    await imageController.initialize();

    elements.imageTab.addEventListener("click", () => setMode("image"));
    elements.plyTab.addEventListener("click", () => setMode("ply"));
    updateTabState("image");
    imageController.setActive(true);
    elements.shell.classList.remove("is-booting");

    window.__orbitViewer = {
      ready: true,
      manifestSummary: {
        pathCount: manifest.stats.path_count,
        sampleCount: manifest.stats.total_camera_samples,
        modelVertexCount: manifest.model.vertex_count,
      },
      setMode,
      getState: () => ({
        mode: currentMode,
        image: imageController?.getState() ?? null,
        ply: plyViewer?.getState() ?? null,
      }),
    };
    window.dispatchEvent(new CustomEvent("orbit-viewer-ready"));
  } catch (error) {
    elements.bootTitle.textContent = "Viewer 無法啟動";
    elements.bootDetail.textContent = error instanceof Error ? error.message : String(error);
    showError(error);
  }
}

window.addEventListener("unhandledrejection", (event) => {
  showError(event.reason ?? new Error("發生未處理的前端錯誤"));
});

start();
