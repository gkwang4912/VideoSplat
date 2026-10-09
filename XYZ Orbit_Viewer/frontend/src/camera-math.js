export const DEG_TO_RAD = Math.PI / 180;
export const RAD_TO_DEG = 180 / Math.PI;

export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function wrapDegrees(value) {
  return ((value + 180) % 360 + 360) % 360 - 180;
}

export function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(vector, scalar) {
  return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function length(vector) {
  return Math.sqrt(dot(vector, vector));
}

export function normalize(vector) {
  const vectorLength = length(vector);
  if (vectorLength < 1e-9) return [0, 0, 0];
  return scale(vector, 1 / vectorLength);
}

function reject(vector, basisVectors) {
  let result = [...vector];
  for (const basis of basisVectors) {
    result = subtract(result, scale(basis, dot(result, basis)));
  }
  return result;
}

export function prepareSamples(paths, subjectCenter) {
  let globalIndex = 0;
  return paths.flatMap((path) =>
    path.frames.map((frame) => ({
      ...frame,
      globalIndex: globalIndex++,
      pathId: path.id,
      startAnchor: path.start_anchor,
      endAnchor: path.end_anchor,
      direction: normalize(subtract(frame.camera_position, subjectCenter)),
    })),
  );
}

export function createOrbitBasis(anchors, subjectCenter) {
  const anchorDirection = (id, fallback) => {
    const anchor = anchors.find((item) => item.id === id);
    return anchor
      ? normalize(subtract(anchor.camera_position, subjectCenter))
      : fallback;
  };

  const front = anchorDirection("front", [0, 0, -1]);
  const rightCandidate = anchorDirection("right", [1, 0, 0]);
  const right = normalize(reject(rightCandidate, [front]));
  const topCandidate = anchorDirection("top", [0, -1, 0]);
  let top = normalize(reject(topCandidate, [front, right]));
  if (dot(top, topCandidate) < 0) top = scale(top, -1);

  return { front, right, top };
}

export function directionFromOrbit(yawDegrees, pitchDegrees, basis) {
  const yaw = yawDegrees * DEG_TO_RAD;
  const pitch = pitchDegrees * DEG_TO_RAD;
  const horizontal = Math.cos(pitch);
  return normalize(
    add(
      add(
        scale(basis.front, horizontal * Math.cos(yaw)),
        scale(basis.right, horizontal * Math.sin(yaw)),
      ),
      scale(basis.top, Math.sin(pitch)),
    ),
  );
}

export function orbitFromDirection(direction, basis) {
  const normalizedDirection = normalize(direction);
  const x = dot(normalizedDirection, basis.right);
  const y = clamp(dot(normalizedDirection, basis.top), -1, 1);
  const z = dot(normalizedDirection, basis.front);
  return {
    yaw: wrapDegrees(Math.atan2(x, z) * RAD_TO_DEG),
    pitch: Math.asin(y) * RAD_TO_DEG,
  };
}

export function angularErrorDegrees(a, b) {
  return Math.acos(clamp(dot(normalize(a), normalize(b)), -1, 1)) * RAD_TO_DEG;
}

export function findNearestSample(samples, direction, previousPathId = null) {
  const target = normalize(direction);
  let bestSample = null;
  let bestDot = Number.NEGATIVE_INFINITY;
  let pathBestSample = null;
  let pathBestDot = Number.NEGATIVE_INFINITY;

  for (const sample of samples) {
    const similarity = dot(sample.direction, target);
    if (similarity > bestDot) {
      bestDot = similarity;
      bestSample = sample;
    }
    if (sample.pathId === previousPathId && similarity > pathBestDot) {
      pathBestDot = similarity;
      pathBestSample = sample;
    }
  }

  // At shared endpoints several paths occupy the same camera position. Keep the
  // current path within a tiny tolerance so the source image does not flicker.
  const continuityAllowance = 0.00004;
  if (pathBestSample && bestDot - pathBestDot <= continuityAllowance) {
    return {
      sample: pathBestSample,
      errorDegrees: Math.acos(clamp(pathBestDot, -1, 1)) * RAD_TO_DEG,
    };
  }

  return {
    sample: bestSample,
    errorDegrees: Math.acos(clamp(bestDot, -1, 1)) * RAD_TO_DEG,
  };
}
