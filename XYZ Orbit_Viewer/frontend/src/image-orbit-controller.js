import {
  clamp,
  createOrbitBasis,
  directionFromOrbit,
  findNearestSample,
  orbitFromDirection,
  prepareSamples,
  wrapDegrees,
} from "./camera-math.js";
import { orbitHeading, pathLabel, shortPathLabel } from "./labels.js";
import { SmartPathOrientation } from "./path-orientation.js";

export class ImageOrbitController {
  constructor(manifest, elements) {
    this.manifest = manifest;
    this.elements = elements;
    this.paths = manifest.paths;
    this.pathMap = new Map(this.paths.map((path) => [path.id, path]));
    this.samples = prepareSamples(this.paths, manifest.subject_center);
    this.basis = createOrbitBasis(manifest.anchors, manifest.subject_center);
    this.config = manifest.config?.image_mode ?? {};
    this.dragDegreesPerPixel = this.config.drag_degrees_per_pixel ?? 0.24;
    this.keyboardStepDegrees = this.config.keyboard_step_degrees ?? 1;
    this.smartPath = new SmartPathOrientation(
      this.paths,
      this.samples,
      manifest.config?.path_playback?.smart_reverse !== false,
    );
    this.yaw = 0;
    this.pitch = 0;
    this.active = true;
    this.currentSample = null;
    this.currentOrientation = null;
    this.currentError = 0;
    this.dragPointerId = null;
    this.dragX = 0;
    this.dragY = 0;
    this.pendingOrbitFrame = 0;
    this.imageRequestToken = 0;
    this.loadedUrls = new Set();
    this.routeButtons = new Map();
  }

  async initialize() {
    this.renderRouteButtons();
    this.bindEvents();
    await this.selectFromOrbit({ waitForImage: true });
  }

  renderRouteButtons() {
    const fragment = document.createDocumentFragment();
    for (const [index, path] of this.paths.entries()) {
      const item = document.createElement("div");
      item.setAttribute("role", "listitem");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "route-button";
      button.dataset.pathId = path.id;
      button.title = pathLabel(path);

      const number = document.createElement("span");
      number.className = "route-number";
      number.textContent = String(index + 1).padStart(2, "0");

      const name = document.createElement("span");
      name.className = "route-name";
      name.textContent = shortPathLabel(path);

      button.append(number, name);
      button.addEventListener("click", () => this.selectPath(path.id));
      this.routeButtons.set(path.id, button);
      item.appendChild(button);
      fragment.appendChild(item);
    }
    this.elements.routeList.replaceChildren(fragment);
  }

  bindEvents() {
    const { stage, slider } = this.elements;

    stage.addEventListener("pointerdown", (event) => {
      if (!this.active || event.target.closest("input, button")) return;
      this.dragPointerId = event.pointerId;
      this.dragX = event.clientX;
      this.dragY = event.clientY;
      stage.setPointerCapture(event.pointerId);
      stage.classList.add("is-dragging");
      stage.focus({ preventScroll: true });
    });

    stage.addEventListener("pointermove", (event) => {
      if (event.pointerId !== this.dragPointerId) return;
      const deltaX = event.clientX - this.dragX;
      const deltaY = event.clientY - this.dragY;
      this.dragX = event.clientX;
      this.dragY = event.clientY;
      this.yaw = wrapDegrees(this.yaw + deltaX * this.dragDegreesPerPixel);
      this.pitch = clamp(
        this.pitch - deltaY * this.dragDegreesPerPixel,
        -89.5,
        89.5,
      );
      this.scheduleOrbitSelection();
    });

    const finishDrag = (event) => {
      if (event.pointerId !== this.dragPointerId) return;
      this.dragPointerId = null;
      stage.classList.remove("is-dragging");
      if (stage.hasPointerCapture(event.pointerId)) {
        stage.releasePointerCapture(event.pointerId);
      }
    };
    stage.addEventListener("pointerup", finishDrag);
    stage.addEventListener("pointercancel", finishDrag);

    slider.addEventListener("input", () => {
      const path = this.pathMap.get(this.currentSample?.pathId ?? this.paths[0].id);
      const displayIndex = clamp(Number(slider.value), 0, path.frames.length - 1);
      const frameIndex = this.currentOrientation?.reversed
        ? path.frames.length - 1 - displayIndex
        : displayIndex;
      const sample = this.samples.find(
        (item) => item.pathId === path.id && item.frame_index === frameIndex,
      );
      if (sample) {
        this.selectSample(sample, 0, {
          syncOrbit: true,
          orientationOverride: this.currentOrientation?.reversed ?? false,
        });
      }
    });

    window.addEventListener("keydown", (event) => {
      if (!this.active || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.target instanceof HTMLInputElement && event.target !== stage) return;
      let handled = true;
      const step = this.keyboardStepDegrees;
      if (event.key === "ArrowLeft") this.yaw = wrapDegrees(this.yaw - step);
      else if (event.key === "ArrowRight") this.yaw = wrapDegrees(this.yaw + step);
      else if (event.key === "ArrowUp") this.pitch = clamp(this.pitch + step, -89.5, 89.5);
      else if (event.key === "ArrowDown") this.pitch = clamp(this.pitch - step, -89.5, 89.5);
      else handled = false;

      if (handled) {
        event.preventDefault();
        this.selectFromOrbit();
      }
    });
  }

  scheduleOrbitSelection() {
    if (this.pendingOrbitFrame) return;
    this.pendingOrbitFrame = requestAnimationFrame(() => {
      this.pendingOrbitFrame = 0;
      this.selectFromOrbit();
    });
  }

  async selectFromOrbit({ waitForImage = false } = {}) {
    const direction = directionFromOrbit(this.yaw, this.pitch, this.basis);
    const match = findNearestSample(
      this.samples,
      direction,
      this.currentSample?.pathId ?? null,
    );
    if (!match.sample) return;
    return this.selectSample(match.sample, match.errorDegrees, {
      syncOrbit: false,
      waitForImage,
      viewDirection: direction,
    });
  }

  selectPath(pathId) {
    const pathSamples = this.samples.filter((sample) => sample.pathId === pathId);
    const direction = directionFromOrbit(this.yaw, this.pitch, this.basis);
    const reversed = this.smartPath.chooseEntry(pathId, direction);
    const match = findNearestSample(pathSamples, direction, null);
    if (match.sample) {
      this.selectSample(match.sample, match.errorDegrees, {
        syncOrbit: true,
        orientationOverride: reversed,
        viewDirection: match.sample.direction,
      });
    }
  }

  async selectSample(
    sample,
    errorDegrees,
    {
      syncOrbit,
      waitForImage = false,
      viewDirection = null,
      orientationOverride = null,
    },
  ) {
    if (syncOrbit) {
      const orbit = orbitFromDirection(sample.direction, this.basis);
      this.yaw = orbit.yaw;
      this.pitch = orbit.pitch;
      errorDegrees = 0;
    }

    const effectiveDirection =
      viewDirection ?? directionFromOrbit(this.yaw, this.pitch, this.basis);
    this.currentOrientation =
      orientationOverride === null
        ? this.smartPath.update(sample, effectiveDirection)
        : this.smartPath.force(sample, effectiveDirection, orientationOverride);

    const changed = this.currentSample?.id !== sample.id;
    this.currentSample = sample;
    this.currentError = errorDegrees;
    this.updateReadouts();

    if (changed) {
      const imagePromise = this.loadImage(sample.image_url);
      this.prefetchNeighbors(sample);
      if (waitForImage) await imagePromise;
    }
  }

  updateReadouts() {
    const sample = this.currentSample;
    if (!sample) return;
    const path = this.pathMap.get(sample.pathId);
    const orientation = this.currentOrientation ?? this.smartPath.describe(sample);
    const frameNumber = orientation.displayFrameIndex + 1;
    const frameTotal = path.frames.length;

    this.elements.heading.textContent = orbitHeading(this.yaw, this.pitch);
    this.elements.coordinates.textContent = `水平 ${Math.round(this.yaw)}° / 垂直 ${Math.round(this.pitch)}°`;
    this.elements.pathLabel.textContent = pathLabel(path, orientation.reversed);
    this.elements.frameAngle.textContent = `${String(Math.round(orientation.displayAngleDegrees)).padStart(3, "0")}°`;
    this.elements.slider.max = String(frameTotal - 1);
    this.elements.slider.value = String(orientation.displayFrameIndex);
    this.elements.slider.style.setProperty(
      "--range-progress",
      `${(orientation.displayFrameIndex / Math.max(1, frameTotal - 1)) * 100}%`,
    );
    this.elements.matchError.textContent = `${this.currentError.toFixed(2)}°`;
    this.elements.frameId.textContent = `${String(frameNumber).padStart(3, "0")} / ${String(frameTotal).padStart(3, "0")}`;
    this.elements.framePath.textContent = sample.pathId;

    for (const [pathId, button] of this.routeButtons) {
      const isActive = pathId === sample.pathId;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
      button.querySelector(".route-name").textContent = shortPathLabel(
        this.pathMap.get(pathId),
        isActive && orientation.reversed,
      );
    }
  }

  async loadImage(url) {
    const token = ++this.imageRequestToken;
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
    if (token !== this.imageRequestToken) return;
    this.elements.image.src = url;
    this.elements.stage.classList.add("is-image-ready");
    this.loadedUrls.add(url);
  }

  prefetchNeighbors(sample) {
    const path = this.pathMap.get(sample.pathId);
    for (const offset of [-2, -1, 1, 2]) {
      const frame = path.frames[sample.frame_index + offset];
      if (!frame || this.loadedUrls.has(frame.image_url)) continue;
      const image = new Image();
      image.decoding = "async";
      image.src = frame.image_url;
      this.loadedUrls.add(frame.image_url);
    }
  }

  setActive(active) {
    this.active = active;
    if (active) this.elements.stage.focus({ preventScroll: true });
  }

  getState() {
    return {
      yaw: this.yaw,
      pitch: this.pitch,
      sampleId: this.currentSample?.id ?? null,
      pathId: this.currentSample?.pathId ?? null,
      frameIndex: this.currentSample?.frame_index ?? null,
      displayFrameIndex: this.currentOrientation?.displayFrameIndex ?? null,
      pathReversed: this.currentOrientation?.reversed ?? false,
      displayStartAnchor: this.currentOrientation?.startAnchor ?? null,
      displayEndAnchor: this.currentOrientation?.endAnchor ?? null,
      matchErrorDegrees: this.currentError,
    };
  }
}
