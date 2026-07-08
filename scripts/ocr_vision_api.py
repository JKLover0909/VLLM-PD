#!/usr/bin/env python3
"""OCR image-based PDFs using Google Cloud Vision API."""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import fitz  # PyMuPDF


SRC = Path("documents/Research/DocJP")
DST = Path("documents/Research/DocJP_md")

EMPTY_FILES = [
    "【法務】3rdWATCH_ユーザマニュアル_第2版.pdf",
    "【法務】3rdWATCH_管理者マニュアル（管理機能）.pdf",
    "【法務】3rdWATCH_管理者マニュアル（起動・集計機能）.pdf",
    "【法務】捺印管理システム＿トップページ.pdf",
    "【総務】元データ‗様式16号の3‗通勤災害用（療養給付たる療養の給付請求書）.pdf",
    "【総務】廃棄物・有価物保管場所配置図.pdf",
]


def ocr_image(image_bytes: bytes, api_key: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["ja", "en"]},
        }]
    }
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            annotations = result["responses"][0].get("fullTextAnnotation")
            return annotations["text"] if annotations else ""
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"    HTTP {e.code}: {body[:200]}")
            if e.code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
    return ""


def process_pdf(pdf_path: Path, api_key: str) -> str:
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages: list[str] = []

    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        text = ocr_image(img_bytes, api_key).strip()
        if text:
            pages.append(f"<!-- page {i} -->\n{text}")
            print(f"    page {i}/{total}: {len(text)} chars")
        else:
            print(f"    page {i}/{total}: (empty)")
        if i < total:
            time.sleep(0.3)

    doc.close()
    return "\n\n".join(pages)


def main() -> None:
    api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    if not api_key:
        print("Usage: python ocr_vision_api.py <GOOGLE_CLOUD_VISION_API_KEY>")
        sys.exit(1)

    DST.mkdir(parents=True, exist_ok=True)
    ok = 0
    empty = 0

    for name in EMPTY_FILES:
        pdf_path = SRC / name
        if not pdf_path.exists():
            print(f"  NOT FOUND: {name}")
            continue

        out_path = DST / (pdf_path.stem + ".md")
        print(f"\n  OCR: {name}")

        try:
            text = process_pdf(pdf_path, api_key)
            if text.strip():
                out_path.write_text(f"# {pdf_path.stem}\n\n{text}\n", encoding="utf-8")
                print(f"  OK  -> {out_path.name} ({len(text)} chars)")
                ok += 1
            else:
                out_path.write_text(
                    f"# {pdf_path.stem}\n\n(OCRでテキストを抽出できませんでした)\n",
                    encoding="utf-8",
                )
                print(f"  EMPTY (no text extracted)")
                empty += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\nDone: {ok} OK, {empty} empty / {len(EMPTY_FILES)} total")


if __name__ == "__main__":
    main()
