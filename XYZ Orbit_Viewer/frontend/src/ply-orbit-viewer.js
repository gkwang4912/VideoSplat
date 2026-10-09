import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

import { findNearestSample, prepareSamples } from "./camera-math.js";
import { pathLabel } from "./labels.js";
import { SmartPathOrientation } from "./path-orientation.js";

export class PlyOrbitViewer {
  constructor(manifest, elements, onError) {
    this.manifest = manifest;
    this.elements = elements;
    this.onError = onError;
    this.samples = prepareSamples(manifest.paths, manifest.subject_center);
    this.pathMap = new Map(manifest.paths.map((path) => [path.id, path]));
    this.center = new THREE.Vector3(...manifest.subject_center);
    this.radius = manifest.sphere_radius;
    this.config = manifest.config?.ply_mode ?? {};
    this.modelScale = this.config.model_scale ?? 1;
    this.cameraOrbitScale = this.config.camera_orbit_scale ?? 1;
    this.visualRadius = this.radius * this.cameraOrbitScale;
    this.cameraColor = new THREE.Color(this.config.camera_point_color ?? "#737d89");
    this.pathColor = new THREE.Color(this.config.path_color ?? "#59626e");
    this.selectedColor = new THREE.Color(
      this.config.selected_camera_color ?? "#ffc857",
    );
    this.smartPath = new SmartPathOrientation(
      manifest.paths,
      this.samples,
      manifest.config?.path_playback?.smart_reverse !== false,
    );
    this.initialized = false;
    this.loaded = false;
    this.active = false;
    this.cameraPathsVisible = true;
    this.currentSample = null;
    this.currentOrientation = null;
    this.currentError = 0;
    this.comparisonRequestToken = 0;
    this.modelBounds = null;
  }

  async initialize() {
    if (this.initialized) return;
    this.initialized = true;

    try {
      this.createThreeScene();
      this.createCameraOverlay();
      this.bindControls();
      this.resize();
      this.renderer.setAnimationLoop(() => this.animate());
      await this.loadSplat();
      this.loaded = true;
      this.elements.loading.classList.add("is-complete");
      this.updateSelectionFromCamera(true);
    } catch (error) {
      this.initialized = false;
      this.elements.loadingTitle.textContent = "PLY 載入失敗";
      this.elements.loadingDetail.textContent = error instanceof Error ? error.message : String(error);
      this.elements.loading.classList.remove("is-complete");
      this.onError(error);
      throw error;
    }
  }

  createThreeScene() {
    const host = this.elements.canvasHost;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080a0c);

    this.camera = new THREE.PerspectiveCamera(
      this.config.overview_fov_degrees ?? 54,
      1,
      0.01,
      Math.max(this.radius, this.visualRadius) * 30,
    );
    const frontAnchor = this.manifest.anchors.find((anchor) => anchor.id === "front");
    const initialPosition = frontAnchor?.camera_position ?? [0, 0, -this.radius];
    const cameraUp = frontAnchor?.camera_up ?? [0, -1, 0];
    const initialDirection = new THREE.Vector3(...initialPosition).sub(this.center).normalize();
    this.camera.position
      .copy(this.center)
      .addScaledVector(
        initialDirection,
        this.visualRadius * (this.config.overview_distance_multiplier ?? 2.55),
      );
    this.camera.up.fromArray(cameraUp).normalize();
    this.camera.lookAt(this.center);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setClearColor(0x080a0c, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute("aria-label", "可旋轉的 Gaussian Splat PLY 場景");
    this.renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.onError(new Error("WebGL 連線中斷，請重新整理頁面"));
    });
    host.appendChild(this.renderer.domElement);

    this.spark = new SparkRenderer({
      renderer: this.renderer,
      sortRadial: true,
      minSortIntervalMs: 16,
      focalAdjustment: 1,
    });
    this.spark.renderOrder = 1;
    this.scene.add(this.spark);

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.center);
    this.controls.enableDamping = !reducedMotion;
    this.controls.dampingFactor = 0.075;
    this.controls.enablePan = false;
    this.controls.rotateSpeed = 0.6;
    this.controls.zoomSpeed = 0.75;
    this.controls.minDistance =
      this.radius * (this.config.min_zoom_distance_multiplier ?? 0.48);
    this.controls.maxDistance =
      Math.max(this.radius, this.visualRadius) *
      (this.config.max_zoom_distance_multiplier ?? 2.6);
    this.controls.update();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
  }

  createCameraOverlay() {
    this.cameraOverlay = new THREE.Group();
    this.cameraOverlay.name = "camera-overlay";
    this.scene.add(this.cameraOverlay);

    const positions = new Float32Array(this.samples.length * 3);
    const colors = new Float32Array(this.samples.length * 3);
    for (const sample of this.samples) {
      const offset = sample.globalIndex * 3;
      positions.set(this.displayCameraPosition(sample.camera_position).toArray(), offset);
      colors.set(this.cameraColor.toArray(), offset);
    }

    this.cameraPointGeometry = new THREE.BufferGeometry();
    this.cameraPointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.cameraColorAttribute = new THREE.BufferAttribute(colors, 3);
    this.cameraPointGeometry.setAttribute("color", this.cameraColorAttribute);

    this.cameraPoints = new THREE.Points(
      this.cameraPointGeometry,
      new THREE.PointsMaterial({
        size: this.config.camera_point_size_px ?? 3,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: this.config.camera_point_opacity ?? 0.8,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.cameraPoints.frustumCulled = false;
    this.cameraPoints.renderOrder = 6;
    this.cameraOverlay.add(this.cameraPoints);

    this.pathLines = new THREE.Group();
    this.pathLines.name = "camera-path-lines";
    for (const path of this.manifest.paths) {
      const geometry = new THREE.BufferGeometry().setFromPoints(
        path.frames.map((frame) => this.displayCameraPosition(frame.camera_position)),
      );
      const material = new THREE.LineBasicMaterial({
        color: this.pathColor,
        transparent: true,
        opacity: this.config.path_opacity ?? 0.42,
        depthTest: false,
        depthWrite: false,
      });
      const line = new THREE.Line(geometry, material);
      line.frustumCulled = false;
      line.renderOrder = 5;
      this.pathLines.add(line);
    }
    this.cameraOverlay.add(this.pathLines);

    this.selectedMarker = new THREE.Mesh(
      new THREE.RingGeometry(0.62, 1, 28),
      new THREE.MeshBasicMaterial({
        color: this.selectedColor,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.96,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.selectedMarker.renderOrder = 8;
    this.selectedMarker.frustumCulled = false;
    this.selectedMarker.visible = false;
    this.cameraOverlay.add(this.selectedMarker);

    this.selectedRayGeometry = new THREE.BufferGeometry().setFromPoints([
      this.center,
      this.center,
    ]);
    this.selectedRay = new THREE.Line(
      this.selectedRayGeometry,
      new THREE.LineBasicMaterial({
        color: this.selectedColor,
        transparent: true,
        opacity: this.config.selected_ray_opacity ?? 0.48,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.selectedRay.renderOrder = 7;
    this.selectedRay.visible = false;
    this.cameraOverlay.add(this.selectedRay);

    this.axes = new THREE.AxesHelper(this.config.axis_length ?? this.radius * 0.5);
    this.axes.position
      .copy(this.center)
      .add(new THREE.Vector3(...(this.config.axis_offset ?? [0, 0, 0])));
    this.axes.renderOrder = 4;
    const axesMaterials = Array.isArray(this.axes.material) ? this.axes.material : [this.axes.material];
    for (const material of axesMaterials) {
      material.depthTest = false;
      material.depthWrite = false;
      material.transparent = true;
      material.opacity = this.config.axis_opacity ?? 0.9;
    }
    this.cameraOverlay.add(this.axes);
  }

  bindControls() {
    const scaleMin = this.config.model_scale_min ?? 0.25;
    const scaleMax = this.config.model_scale_max ?? 4;
    const scaleStep = this.config.model_scale_step ?? 0.05;
    this.elements.modelScaleInput.min = String(scaleMin);
    this.elements.modelScaleInput.max = String(scaleMax);
    this.elements.modelScaleInput.step = String(scaleStep);
    this.setModelScale(this.modelScale);
    this.elements.modelScaleInput.addEventListener("input", () => {
      this.setModelScale(Number(this.elements.modelScaleInput.value));
    });
    this.elements.resetButton.addEventListener("click", () => this.resetView());
    this.elements.focusModelButton.addEventListener("click", () => this.focusModel());
    this.elements.toggleCamerasButton.addEventListener("click", () => {
      this.cameraPathsVisible = !this.cameraPathsVisible;
      this.cameraPoints.visible = this.cameraPathsVisible;
      this.pathLines.visible = this.cameraPathsVisible;
      this.axes.visible = this.cameraPathsVisible;
      this.elements.toggleCamerasButton.setAttribute(
        "aria-pressed",
        String(this.cameraPathsVisible),
      );
      this.elements.toggleCamerasButton.textContent = this.cameraPathsVisible
        ? "隱藏相機路徑"
        : "顯示相機路徑";
    });
    this.elements.toggleCamerasButton.textContent = "隱藏相機路徑";
  }

  async loadSplat() {
    this.elements.loadingTitle.textContent = "載入 Gaussian Splat";
    this.elements.loadingDetail.textContent = `${this.manifest.model.name} / ${this.formatBytes(this.manifest.model.size_bytes)}`;

    this.splat = new SplatMesh({
      url: this.manifest.model.url,
      onProgress: (event) => {
        const total = event.total || this.manifest.model.size_bytes;
        const ratio = total > 0 ? Math.min(1, event.loaded / total) : 0;
        this.elements.loadBar.style.width = `${ratio * 100}%`;
        this.elements.loadingDetail.textContent = `${Math.round(ratio * 100)}% / ${this.formatBytes(total)}`;
      },
    });
    this.splat.name = "subject-gaussian-splat";
    this.applyModelScale();
    this.scene.add(this.splat);
    await this.splat.initialized;
    this.modelBounds = this.splat.getBoundingBox();
    this.elements.loadBar.style.width = "100%";
    this.elements.loadingTitle.textContent = "PLY 已載入";
    this.elements.loadingDetail.textContent = `${this.manifest.model.vertex_count.toLocaleString()} splats`;
  }

  animate() {
    if (!this.active || !this.renderer) return;
    this.controls.update();
    this.updateSelectionFromCamera();
    this.updateMarkerAppearance();
    this.renderer.render(this.scene, this.camera);
  }

  updateMarkerAppearance() {
    if (!this.currentSample) return;
    this.selectedMarker.quaternion.copy(this.camera.quaternion);
    const distance = this.camera.position.distanceTo(this.selectedMarker.position);
    const viewportHeight = Math.max(1, this.renderer.domElement.clientHeight);
    const worldHeight = 2 * distance * Math.tan((this.camera.fov * Math.PI) / 360);
    const markerRadius =
      (worldHeight / viewportHeight) *
      ((this.config.selected_camera_marker_size_px ?? 26) / 2);
    this.selectedMarker.scale.setScalar(markerRadius);
  }

  updateSelectionFromCamera(force = false) {
    if (!this.camera) return;
    const cameraDirection = this.camera.position.clone().sub(this.center).normalize();
    const match = findNearestSample(
      this.samples,
      cameraDirection.toArray(),
      this.currentSample?.pathId ?? null,
    );
    if (!match.sample) return;
    this.currentError = match.errorDegrees;
    const orientation = this.smartPath.update(match.sample, cameraDirection.toArray());
    if (
      !force &&
      match.sample.id === this.currentSample?.id &&
      orientation.reversed === this.currentOrientation?.reversed
    ) {
      this.elements.matchError.textContent = `偏差 ${match.errorDegrees.toFixed(2)}°`;
      return;
    }
    this.selectSample(match.sample, match.errorDegrees, orientation);
  }

  selectSample(sample, errorDegrees, orientation) {
    const previous = this.currentSample;
    this.currentSample = sample;
    this.currentOrientation = orientation;
    this.currentError = errorDegrees;

    if (previous) {
      this.cameraColorAttribute.setXYZ(
        previous.globalIndex,
        this.cameraColor.r,
        this.cameraColor.g,
        this.cameraColor.b,
      );
    }
    this.cameraColorAttribute.setXYZ(
      sample.globalIndex,
      this.selectedColor.r,
      this.selectedColor.g,
      this.selectedColor.b,
    );
    this.cameraColorAttribute.needsUpdate = true;

    const displayPosition = this.displayCameraPosition(sample.camera_position);
    this.selectedMarker.position.copy(displayPosition);
    this.selectedMarker.visible = true;
    this.selectedRay.visible = true;
    this.updateMarkerAppearance();
    const rayPositions = this.selectedRayGeometry.getAttribute("position");
    rayPositions.setXYZ(0, this.center.x, this.center.y, this.center.z);
    rayPositions.setXYZ(1, displayPosition.x, displayPosition.y, displayPosition.z);
    rayPositions.needsUpdate = true;

    const path = this.pathMap.get(sample.pathId);
    this.elements.pathLabel.textContent = pathLabel(path, orientation.reversed);
    this.elements.matchError.textContent = `偏差 ${errorDegrees.toFixed(2)}°`;
    this.elements.frameId.textContent = String(orientation.displayFrameIndex).padStart(3, "0");
    this.elements.frameAngle.textContent = `${Math.round(orientation.displayAngleDegrees)}°`;
    this.elements.cameraPosition.textContent = sample.camera_position
      .map((value) => value.toFixed(2))
      .join(" / ");
    this.loadComparisonImage(sample.image_url);
  }

  async loadComparisonImage(url) {
    const token = ++this.comparisonRequestToken;
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    try {
      await image.decode();
    } catch {
      await new Promise((resolve, reject) => {
        image.addEventListener("load", resolve, { once: true });
        image.addEventListener("error", reject, { once: true });
      });
    }
    if (token !== this.comparisonRequestToken) return;
    this.elements.comparisonImage.src = url;
    this.elements.comparisonImageWrap.classList.add("is-image-ready");
  }

  resetView() {
    const frontAnchor = this.manifest.anchors.find((anchor) => anchor.id === "front");
    const anchorPosition = new THREE.Vector3(
      ...(frontAnchor?.camera_position ?? [0, 0, -this.radius]),
    );
    const direction = anchorPosition.sub(this.center).normalize();
    this.camera.position
      .copy(this.center)
      .addScaledVector(
        direction,
        this.visualRadius * (this.config.overview_distance_multiplier ?? 2.55),
      );
    this.camera.fov = this.config.overview_fov_degrees ?? 54;
    this.camera.updateProjectionMatrix();
    this.camera.up.fromArray(frontAnchor?.camera_up ?? [0, -1, 0]).normalize();
    this.controls.target.copy(this.center);
    this.controls.update();
    this.updateSelectionFromCamera(true);
  }

  focusModel() {
    const direction = this.camera.position.clone().sub(this.center).normalize();
    this.camera.position
      .copy(this.center)
      .addScaledVector(
        direction,
        this.radius * (this.config.focus_distance_multiplier ?? 1.18),
      );
    this.camera.fov = this.config.focus_fov_degrees ?? 30;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(this.center);
    this.controls.update();
    this.updateSelectionFromCamera(true);
  }

  setModelScale(scale) {
    const min = Number(this.elements.modelScaleInput.min);
    const max = Number(this.elements.modelScaleInput.max);
    this.modelScale = THREE.MathUtils.clamp(scale, min, max);
    this.elements.modelScaleInput.value = String(this.modelScale);
    this.updateModelScaleReadout();
    this.applyModelScale();
  }

  applyModelScale() {
    if (!this.splat) return;
    this.splat.scale.setScalar(this.modelScale);
    // PLY vertices use world-space coordinates. This translation keeps the
    // subject center fixed while only the character grows or shrinks.
    this.splat.position.copy(this.center).multiplyScalar(1 - this.modelScale);
    this.splat.updateMatrixWorld(true);
  }

  updateModelScaleReadout() {
    this.elements.modelScaleValue.value = `${this.modelScale.toFixed(2)}×`;
    this.elements.modelScaleValue.textContent = `${this.modelScale.toFixed(2)}×`;
  }

  resize() {
    if (!this.renderer || !this.camera) return;
    const width = Math.max(1, this.elements.canvasHost.clientWidth);
    const height = Math.max(1, this.elements.canvasHost.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  setActive(active) {
    this.active = active;
    if (active && this.initialized) {
      requestAnimationFrame(() => {
        this.resize();
        this.updateSelectionFromCamera(true);
      });
    }
  }

  formatBytes(bytes) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  displayCameraPosition(sourcePosition) {
    return new THREE.Vector3(...sourcePosition)
      .sub(this.center)
      .multiplyScalar(this.cameraOrbitScale)
      .add(this.center);
  }

  getState() {
    return {
      initialized: this.initialized,
      loaded: this.loaded,
      sampleId: this.currentSample?.id ?? null,
      pathId: this.currentSample?.pathId ?? null,
      frameIndex: this.currentSample?.frame_index ?? null,
      displayFrameIndex: this.currentOrientation?.displayFrameIndex ?? null,
      pathReversed: this.currentOrientation?.reversed ?? false,
      displayStartAnchor: this.currentOrientation?.startAnchor ?? null,
      displayEndAnchor: this.currentOrientation?.endAnchor ?? null,
      matchErrorDegrees: this.currentError,
      cameraPointCount: this.samples.length,
      modelScale: this.modelScale,
      canvasSize: this.renderer
        ? [this.renderer.domElement.width, this.renderer.domElement.height]
        : [0, 0],
      modelBounds: this.modelBounds
        ? {
            min: this.modelBounds.min.toArray(),
            max: this.modelBounds.max.toArray(),
          }
        : null,
      config: this.config,
    };
  }
}
