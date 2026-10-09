const ANCHOR_LABELS = {
  front: "前方",
  back: "後方",
  left: "左側",
  right: "右側",
  top: "上方",
  bottom: "下方",
};

export function anchorLabel(anchor) {
  return ANCHOR_LABELS[anchor] ?? anchor;
}

function pathAnchors(path, reversed = false) {
  const start = path.start_anchor ?? path.startAnchor;
  const end = path.end_anchor ?? path.endAnchor;
  return reversed ? [end, start] : [start, end];
}

export function pathLabel(path, reversed = false) {
  const [start, end] = pathAnchors(path, reversed);
  return `${anchorLabel(start)}至${anchorLabel(end)}`;
}

export function shortPathLabel(path, reversed = false) {
  const [start, end] = pathAnchors(path, reversed);
  return `${anchorLabel(start).replace("方", "").replace("側", "")} → ${anchorLabel(end).replace("方", "").replace("側", "")}`;
}

export function orbitHeading(yaw, pitch) {
  if (pitch >= 58) return "上方視角";
  if (pitch <= -58) return "下方視角";

  const normalizedYaw = ((yaw % 360) + 360) % 360;
  if (normalizedYaw >= 45 && normalizedYaw < 135) return "右側視角";
  if (normalizedYaw >= 135 && normalizedYaw < 225) return "後方視角";
  if (normalizedYaw >= 225 && normalizedYaw < 315) return "左側視角";
  return "正面視角";
}
