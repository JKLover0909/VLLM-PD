import test from "node:test";
import assert from "node:assert/strict";
import { THEME_DEFAULT, resolveStoredTheme } from "../src/theme-preference.js";

test("factory default is the light theme", () => {
  assert.equal(THEME_DEFAULT, "light");
  assert.equal(resolveStoredTheme(null, null).theme, "light");
  assert.equal(resolveStoredTheme("garbage", null).theme, "light");
});

test("an old implicit 'system' value migrates to light exactly once", () => {
  const first = resolveStoredTheme("system", null);
  assert.deepEqual(first, { theme: "light", migrate: true });
  // After the migration flag is written, a deliberate 'system' choice is respected.
  assert.deepEqual(resolveStoredTheme("system", "1"), { theme: "system", migrate: false });
});

test("explicit light or dark choices are never overridden", () => {
  assert.deepEqual(resolveStoredTheme("dark", null), { theme: "dark", migrate: true });
  assert.deepEqual(resolveStoredTheme("light", null), { theme: "light", migrate: true });
  assert.deepEqual(resolveStoredTheme("dark", "1"), { theme: "dark", migrate: false });
});
