import assert from "node:assert/strict";
import test from "node:test";

import {
  shouldClearEmployeeAuth,
  visibleModeKeys,
  isCctvaiAvailable,
  cctvaiHealthBannerVisibility,
} from "../src/mode-tabs.js";

test("shows QA only when enabled and keeps it visible during agent outages", () => {
  assert.deepEqual(visibleModeKeys(), ["mkac", "mes", "research"]);
  assert.deepEqual(visibleModeKeys({ qa: true }), ["mkac", "mes", "qa", "research"]);
  assert.deepEqual(visibleModeKeys({ qa: true, wms: true }), ["mkac", "mes", "wms", "qa", "research"]);
});

test("hides WMS and CCTVAI until available", () => {
  assert.deepEqual(visibleModeKeys(), ["mkac", "mes", "research"]);
  assert.deepEqual(visibleModeKeys({ wms: true }), ["mkac", "mes", "wms", "research"]);
  assert.deepEqual(visibleModeKeys({ cctvai: true }), ["mkac", "mes", "cctvai", "research"]);
  assert.deepEqual(visibleModeKeys({ wms: true, cctvai: true }), ["mkac", "mes", "wms", "cctvai", "research"]);
});

test("cctvai is available when either database or hardware is available", () => {
  assert.equal(isCctvaiAvailable(), false);
  assert.equal(
    isCctvaiAvailable({ database: { available: false }, hardware: { available: false } }),
    false,
  );
  assert.equal(
    isCctvaiAvailable({ database: { available: true }, hardware: { available: false } }),
    true,
  );
  assert.equal(
    isCctvaiAvailable({ database: { available: false }, hardware: { available: true } }),
    true,
  );
  assert.equal(
    isCctvaiAvailable({ database: { available: true }, hardware: { available: true } }),
    true,
  );
});

test("keeps the CCTVAI tab visible and does not force switching when only hardware is up", () => {
  // Database xuống hoàn toàn nhưng hardware vẫn khả dụng: tab CCTVAI phải
  // vẫn hiện ra trong visibleModeKeys() và mode hiện tại không bị coi là
  // cần auto-switch (isCctvaiAvailable() vẫn true).
  const database = { available: false, enabled: true, state: "UNAVAILABLE" };
  const hardware = { available: true, enabled: true, state: "AVAILABLE" };
  const available = isCctvaiAvailable({ database, hardware });
  assert.equal(available, true);
  assert.deepEqual(visibleModeKeys({ cctvai: available }), [
    "mkac",
    "mes",
    "cctvai",
    "research",
  ]);
});

test("hides the CCTVAI tab only when both database and hardware are unavailable", () => {
  const database = { available: false, enabled: true, state: "UNAVAILABLE" };
  const hardware = { available: false, enabled: false, state: "UNAVAILABLE" };
  const available = isCctvaiAvailable({ database, hardware });
  assert.equal(available, false);
  assert.deepEqual(visibleModeKeys({ cctvai: available }), ["mkac", "mes", "research"]);
});

test("keeps the database outage warning accurate regardless of hardware status", () => {
  // Database down, hardware up: chỉ banner DB phải hiện, banner hardware ẩn.
  assert.deepEqual(
    cctvaiHealthBannerVisibility({
      database: { available: false },
      hardware: { enabled: true, available: true },
    }),
    { showDatabaseBanner: true, showHardwareBanner: false },
  );
  // Ngược lại: database up, hardware down thì chỉ banner hardware hiện.
  assert.deepEqual(
    cctvaiHealthBannerVisibility({
      database: { available: true },
      hardware: { enabled: true, available: false },
    }),
    { showDatabaseBanner: false, showHardwareBanner: true },
  );
  // Cả hai đều lên thì không banner nào hiện.
  assert.deepEqual(
    cctvaiHealthBannerVisibility({
      database: { available: true },
      hardware: { enabled: true, available: true },
    }),
    { showDatabaseBanner: false, showHardwareBanner: false },
  );
  // Cả hai đều xuống: cả hai banner phải hiện.
  assert.deepEqual(
    cctvaiHealthBannerVisibility({
      database: { available: false },
      hardware: { enabled: true, available: false },
    }),
    { showDatabaseBanner: true, showHardwareBanner: true },
  );
});

test("does not warn about hardware monitoring that is switched off", () => {
  // Mặc định CCTVAI_HARDWARE_ENABLED=false: tính năng chưa bật không phải sự
  // cố, nên mode CCTVAI không được kèm banner lỗi vĩnh viễn.
  assert.deepEqual(
    cctvaiHealthBannerVisibility({
      database: { available: true },
      hardware: { enabled: false, available: false },
    }),
    { showDatabaseBanner: false, showHardwareBanner: false },
  );
  // Backend chưa trả field cctvai_hardware → frontend thấy {} → vẫn im lặng.
  assert.deepEqual(
    cctvaiHealthBannerVisibility({ database: { available: true }, hardware: {} }),
    { showDatabaseBanner: false, showHardwareBanner: false },
  );
});

test("clears stale employee auth for protected query modes", () => {
  for (const mode of ["mkac", "mes", "wms", "cctvai", "qa"]) {
    assert.equal(
      shouldClearEmployeeAuth({
        status: 403,
        errorCode: "INVALID_EMPLOYEE_ID",
        mode,
      }),
      true,
    );
  }
});

test("keeps employee auth for ownership errors and public modes", () => {
  assert.equal(
    shouldClearEmployeeAuth({
      status: 403,
      errorCode: "ARTIFACT_FORBIDDEN",
      mode: "wms",
    }),
    false,
  );
  assert.equal(
    shouldClearEmployeeAuth({
      status: 403,
      errorCode: "INVALID_EMPLOYEE_ID",
      mode: "research",
    }),
    false,
  );
});
