const EMPLOYEE_PROTECTED_MODES = new Set(["mkac", "mes", "wms"]);

export function visibleModeKeys(wmsAvailable) {
  return wmsAvailable
    ? ["mkac", "mes", "wms", "research"]
    : ["mkac", "mes", "research"];
}

export function shouldClearEmployeeAuth({ status, errorCode, mode }) {
  return (
    status === 403 &&
    errorCode === "INVALID_EMPLOYEE_ID" &&
    EMPLOYEE_PROTECTED_MODES.has(mode)
  );
}

export function getModeTabScrollLeft({
  scrollLeft,
  clientWidth,
  scrollWidth,
  tabLeft,
  tabWidth,
}) {
  const maxScrollLeft = Math.max(0, scrollWidth - clientWidth);
  const visibleLeft = scrollLeft;
  const visibleRight = scrollLeft + clientWidth;
  let nextScrollLeft = scrollLeft;

  if (tabLeft < visibleLeft) {
    nextScrollLeft = tabLeft;
  } else if (tabLeft + tabWidth > visibleRight) {
    nextScrollLeft = tabLeft + tabWidth - clientWidth;
  }

  return Math.min(maxScrollLeft, Math.max(0, nextScrollLeft));
}
