import { dot, length, normalize, subtract } from "./camera-math.js";

export class SmartPathOrientation {
  constructor(paths, samples, enabled = true) {
    this.enabled = enabled;
    this.pathMap = new Map(paths.map((path) => [path.id, path]));
    this.pathSamples = new Map();
    for (const path of paths) {
      this.pathSamples.set(
        path.id,
        samples
          .filter((sample) => sample.pathId === path.id)
          .sort((a, b) => a.frame_index - b.frame_index),
      );
    }
    this.currentPathId = null;
    this.reversed = false;
    this.previousSample = null;
    this.previousDirection = null;
  }

  chooseEntry(pathId, viewDirection) {
    if (!this.enabled) return false;
    const samples = this.pathSamples.get(pathId) ?? [];
    if (samples.length < 2) return false;
    const direction = normalize(viewDirection);
    const startSimilarity = dot(direction, samples[0].direction);
    const endSimilarity = dot(direction, samples[samples.length - 1].direction);
    return endSimilarity > startSimilarity;
  }

  force(sample, viewDirection, reversed) {
    this.currentPathId = sample.pathId;
    this.reversed = this.enabled ? Boolean(reversed) : false;
    this.previousSample = sample;
    this.previousDirection = normalize(viewDirection);
    return this.describe(sample);
  }

  update(sample, viewDirection) {
    const direction = normalize(viewDirection);
    if (!this.enabled) {
      this.reversed = false;
    } else if (
      this.previousSample &&
      this.previousSample.pathId === sample.pathId &&
      this.currentPathId === sample.pathId
    ) {
      const frameDelta = sample.frame_index - this.previousSample.frame_index;
      if (frameDelta > 0) this.reversed = false;
      else if (frameDelta < 0) this.reversed = true;
      else this.inferFromMovement(sample, direction);
    } else {
      const inferred = this.inferFromMovement(sample, direction);
      if (!inferred) {
        const samples = this.pathSamples.get(sample.pathId) ?? [];
        this.reversed = sample.frame_index > (samples.length - 1) / 2;
      }
    }

    this.currentPathId = sample.pathId;
    this.previousSample = sample;
    this.previousDirection = direction;
    return this.describe(sample);
  }

  inferFromMovement(sample, direction) {
    if (!this.previousDirection) return false;
    const movement = subtract(direction, this.previousDirection);
    if (length(movement) < 1e-6) return false;
    const tangent = this.forwardTangent(sample);
    const score = dot(normalize(movement), tangent);
    if (Math.abs(score) < 0.04) return false;
    this.reversed = score < 0;
    return true;
  }

  forwardTangent(sample) {
    const samples = this.pathSamples.get(sample.pathId) ?? [];
    if (samples.length < 2) return [0, 0, 0];
    const index = Math.max(0, Math.min(samples.length - 1, sample.frame_index));
    const before = samples[Math.max(0, index - 1)];
    const after = samples[Math.min(samples.length - 1, index + 1)];
    return normalize(subtract(after.direction, before.direction));
  }

  describe(sample) {
    const path = this.pathMap.get(sample.pathId);
    const samples = this.pathSamples.get(sample.pathId) ?? [];
    const lastIndex = Math.max(0, samples.length - 1);
    const firstAngle = samples[0]?.angle_degrees ?? 0;
    const lastAngle = samples[lastIndex]?.angle_degrees ?? lastIndex;
    const rawAngle = sample.angle_degrees ?? sample.frame_index;
    const reversed = this.enabled && this.reversed;
    return {
      reversed,
      displayFrameIndex: reversed ? lastIndex - sample.frame_index : sample.frame_index,
      displayAngleDegrees: reversed ? lastAngle - rawAngle : rawAngle - firstAngle,
      startAnchor: reversed ? path.end_anchor : path.start_anchor,
      endAnchor: reversed ? path.start_anchor : path.end_anchor,
      sourceFrameIndex: sample.frame_index,
      frameCount: samples.length,
    };
  }
}
