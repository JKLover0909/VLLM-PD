"""Re-root preview image paths stored during indexing.

`DocumentParser` lưu đường dẫn ảnh trang dưới dạng tuyệt đối tại thời điểm
index (``str(target.resolve())``). Khi index chạy trên host, path là
``/home/.../docjp_processed/...``; nhưng app phục vụ lại chạy trong Docker với
cùng thư mục được mount tại ``/app/docjp_processed/...``. Đường dẫn tuyệt đối cũ
không tồn tại trong container nên ``Path(stored).is_file()`` trả về False.

Helper này tách phần path kể từ segment marker (``docjp_processed``,
``mkac_processed``, ``uploads``) rồi gắn lại dưới ``Path.cwd()`` hiện tại, nên
hoạt động bất kể index chạy ở đâu.
"""

from pathlib import Path
from typing import Optional

# Segment gốc của các thư mục chứa ảnh/preview được index.
_PROCESSED_MARKERS = ("docjp_processed", "mkac_processed", "uploads")


def resolve_processed_image_path(stored_path: Optional[str]) -> Optional[Path]:
    """Return a runtime-valid path for a stored preview image, or None.

    - Nếu path lưu sẵn đã tồn tại (index và serve cùng môi trường), dùng luôn.
    - Nếu không, re-root theo marker segment dưới ``Path.cwd()``.
    """
    if not stored_path:
        return None

    raw = Path(stored_path)
    if raw.is_file():
        return raw

    parts = raw.parts
    for marker in _PROCESSED_MARKERS:
        if marker in parts:
            idx = parts.index(marker)
            candidate = Path.cwd().joinpath(*parts[idx:])
            if candidate.is_file():
                return candidate
            return candidate  # trả candidate để caller kiểm tra/log tiếp
    return raw
