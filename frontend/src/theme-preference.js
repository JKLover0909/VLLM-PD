// Factory floors are brightly lit, so the app defaults to the light theme.
// Earlier builds defaulted to "system" and stored that value without the user
// choosing it, so a one-time migration turns a stored "system" into "light".
// After that, any choice the user makes (including "system") is kept.

export const THEME_DEFAULT = "light";
export const THEME_MIGRATION_KEY = "meibook-theme-light-default";
const VALID = new Set(["system", "light", "dark"]);

export function resolveStoredTheme(stored, migrated) {
  const done = migrated === "1";
  if (!VALID.has(stored)) return { theme: THEME_DEFAULT, migrate: !done };
  if (!done && stored === "system") return { theme: THEME_DEFAULT, migrate: true };
  return { theme: stored, migrate: !done };
}
