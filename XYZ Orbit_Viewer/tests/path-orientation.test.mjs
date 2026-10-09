import assert from "node:assert/strict";
import test from "node:test";

import { SmartPathOrientation } from "../frontend/src/path-orientation.js";

const paths = [
  {
    id: "back_to_top",
    start_anchor: "back",
    end_anchor: "top",
  },
];

const samples = [
  { pathId: "back_to_top", frame_index: 0, angle_degrees: 0, direction: [0, 0, 1] },
  { pathId: "back_to_top", frame_index: 1, angle_degrees: 45, direction: [0, -0.707, 0.707] },
  { pathId: "back_to_top", frame_index: 2, angle_degrees: 90, direction: [0, -1, 0] },
];

test("entering from the stored end reverses the path without changing direction", () => {
  const tracker = new SmartPathOrientation(paths, samples, true);
  const viewDirection = [0, -1, 0];
  assert.equal(tracker.chooseEntry("back_to_top", viewDirection), true);
  const state = tracker.force(samples[2], viewDirection, true);
  assert.equal(state.reversed, true);
  assert.equal(state.startAnchor, "top");
  assert.equal(state.endAnchor, "back");
  assert.equal(state.displayFrameIndex, 0);
  assert.deepEqual(viewDirection, [0, -1, 0]);
});

test("moving backward through source frames keeps displayed playback increasing", () => {
  const tracker = new SmartPathOrientation(paths, samples, true);
  tracker.force(samples[2], samples[2].direction, true);
  const middle = tracker.update(samples[1], samples[1].direction);
  const end = tracker.update(samples[0], samples[0].direction);
  assert.equal(middle.reversed, true);
  assert.equal(middle.displayFrameIndex, 1);
  assert.equal(end.reversed, true);
  assert.equal(end.displayFrameIndex, 2);
  assert.equal(end.displayAngleDegrees, 90);
});

test("forward traversal preserves source order", () => {
  const tracker = new SmartPathOrientation(paths, samples, true);
  tracker.force(samples[0], samples[0].direction, false);
  const middle = tracker.update(samples[1], samples[1].direction);
  assert.equal(middle.reversed, false);
  assert.equal(middle.displayFrameIndex, 1);
  assert.equal(middle.startAnchor, "back");
  assert.equal(middle.endAnchor, "top");
});
