const EMPLOYEE_PROTECTED_MODES = new Set(["mkac", "mes", "wms", "cctvai"]);

export function visibleModeKeys({ wms = false, cctvai = false } = {}) {
  const keys = ["mkac", "mes"];
  if (wms) keys.push("wms");
  if (cctvai) keys.push("cctvai");
  keys.push("research");
  return keys;
}

// CCTVAI có hai nguồn dữ liệu độc lập: database report replica (Postgres) và
// hardware monitoring (server metrics). Tab CCTVAI phải giữ hiển thị và
// không tự chuyển mode khỏi nó khi CHỈ MỘT trong hai nguồn khả dụng — mất
// database không được kéo theo mất luôn hardware Q&A và ngược lại.
export function isCctvaiAvailable({ database, hardware } = {}) {
  return Boolean(database?.available) || Boolean(hardware?.available);
}

// Cảnh báo DB và hardware phải tách biệt: banner DB chỉ phản ánh đúng trạng
// thái database (không bị hardware "che" thành available), và ngược lại.
//
// Banner database giữ nguyên điều kiện cũ (!available) để không đổi hành vi
// đã có. Banner hardware chỉ hiện khi tính năng ĐANG BẬT mà không truy cập
// được — hardware mặc định tắt (CCTVAI_HARDWARE_ENABLED=false) nên nếu không
// lọc theo `enabled`, mode CCTVAI sẽ luôn kèm một banner lỗi vĩnh viễn.
export function cctvaiHealthBannerVisibility({ database, hardware } = {}) {
  return {
    showDatabaseBanner: !database?.available,
    showHardwareBanner: Boolean(hardware?.enabled) && !hardware?.available,
  };
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
