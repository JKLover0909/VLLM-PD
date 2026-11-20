import test from "node:test";
import assert from "node:assert/strict";
import { chooseRobotTarget, placeRobot, robotAnchorSelectors, robotHeight, travelDuration } from "../src/robot-path.js";

const base = { state: "ready", hasConversation: false, hasError: false, previewOpen: false, hoverIndex: null, patrolIndex: null, messageVisible: true };

test("welcome screen: home by default, follows hover and patrol", () => {
  assert.equal(chooseRobotTarget(base), "home");
  assert.equal(chooseRobotTarget({ ...base, hoverIndex: 2 }), "card:2");
  assert.equal(chooseRobotTarget({ ...base, patrolIndex: 1 }), "card:1");
  // Real user hover wins over the idle patrol.
  assert.equal(chooseRobotTarget({ ...base, patrolIndex: 1, hoverIndex: 0 }), "card:0");
  assert.equal(chooseRobotTarget({ ...base, previewOpen: true, hoverIndex: 0 }), "preview");
});

test("real request state outranks user hover", () => {
  assert.equal(chooseRobotTarget({ ...base, state: "working", hasConversation: true, hoverIndex: 1 }), "message");
  assert.equal(chooseRobotTarget({ ...base, state: "answering", hasConversation: true, messageVisible: false }), "corner");
});

test("errors send the robot to the error banner when one is shown", () => {
  assert.equal(chooseRobotTarget({ ...base, hasError: true, state: "working", hasConversation: true }), "error");
  assert.equal(chooseRobotTarget({ ...base, state: "error", hasConversation: true }), "message");
  assert.equal(chooseRobotTarget({ ...base, state: "error" }), "home");
});

test("conversation: sits by the latest answer, or the bar if it is scrolled away", () => {
  assert.equal(chooseRobotTarget({ ...base, hasConversation: true }), "message");
  assert.equal(chooseRobotTarget({ ...base, hasConversation: true, messageVisible: false }), "corner");
  assert.equal(chooseRobotTarget({ ...base, hasConversation: true, state: "composing" }), "composer");
  // Patrol never runs once a conversation exists.
  assert.equal(chooseRobotTarget({ ...base, hasConversation: true, patrolIndex: 0 }), "message");
});

test("robotAnchorSelectors scopes error to latest alert and keeps error-banner first", () => {
  assert.deepEqual(robotAnchorSelectors("error"), [".error-banner", '.message[data-latest="true"] .msm-error[role="alert"]']);
});

test("robotAnchorSelectors keeps existing target selectors and rejects unknown targets", () => {
  assert.deepEqual(robotAnchorSelectors("home"), [".agent-presence.stage [data-robot-anchor]"]);
  assert.deepEqual(robotAnchorSelectors("corner"), [".conversation-scroll"]);
  assert.deepEqual(robotAnchorSelectors("preview"), [".task-preview"]);
  assert.deepEqual(robotAnchorSelectors("composer"), [".composer"]);
  assert.deepEqual(robotAnchorSelectors("message"), ['.message[data-latest="true"] .message-avatar']);
  assert.deepEqual(robotAnchorSelectors("card:2"), ['.task-card[data-task-index="2"]']);
  assert.deepEqual(robotAnchorSelectors("unknown"), []);
});

const container = { left: 100, top: 50, width: 1000, height: 700, right: 1100, bottom: 750 };

test("placement is relative to the container and clamped inside it", () => {
  const card = { left: 300, top: 300, width: 200, height: 100, right: 500, bottom: 400 };
  const p = placeRobot("card", card, container, 64);
  assert.equal(Math.round(p.x), Math.round(500 - 64 * 0.62 - 100));
  assert.ok(p.y < card.top - container.top);
  const corner = placeRobot("card", { left: 1090, top: 40, width: 10, height: 10, right: 1100, bottom: 50 }, container, 64);
  assert.ok(corner.x <= container.width - 64 - 4);
  assert.ok(corner.y >= 4);
});

test("home and bar centre the robot on the anchor", () => {
  const anchor = { left: 500, top: 100, width: 132, height: 126, right: 632, bottom: 226 };
  const p = placeRobot("home", anchor, container, 132);
  assert.equal(p.x, 400);
  assert.equal(Math.round(p.y), Math.round(100 + 63 - robotHeight(132) / 2 - 50));
});

test("preview: beside the box when there is room, above it otherwise", () => {
  const wide = placeRobot("preview", { left: 400, top: 300, width: 400, height: 150, right: 800, bottom: 450 }, container, 64);
  assert.ok(wide.x + 64 <= 400 - 100);
  const narrow = placeRobot("preview", { left: 110, top: 300, width: 300, height: 150, right: 410, bottom: 450 }, container, 46);
  assert.ok(narrow.y < 300 - 50);
});

test("invalid rectangles are rejected", () => {
  assert.equal(placeRobot("card", null, container, 64), null);
  assert.equal(placeRobot("card", { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 }, container, 64), null);
});

test("travel time grows with distance and stays bounded", () => {
  assert.equal(travelDuration(0, 0, 0, 0), 450);
  assert.ok(travelDuration(0, 0, 300, 0) > 450);
  assert.equal(travelDuration(0, 0, 5000, 5000), 1100);
});

test("corner: docks inside the conversation area's top-right corner", () => {
  const scroller = { left: 100, top: 120, width: 1000, height: 600, right: 1100, bottom: 720 };
  const p = placeRobot("corner", scroller, container, 56);
  assert.equal(Math.round(p.x), Math.round(1100 - 56 - 16 - 100));
  assert.equal(Math.round(p.y), Math.round(120 + 10 - 50));
});
