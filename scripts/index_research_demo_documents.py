#!/usr/bin/env python3
"""Index prepared Research demo documents into a fixed Qdrant session."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.embedder import Embedder
from src.rag.parser import DocumentParser
from src.rag.vector_store import VectorStore

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}
DEFAULT_RESEARCH_DEMO_SESSION_ID = "00000000-0000-4000-8000-000000000001"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prune", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise SystemExit(f"Research demo source directory not found: {source_dir}")
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def build_embedding_text(chunk, metadata: dict) -> str:
    return (
        f"Không gian: Nghiên cứu tài liệu\n"
        f"Tài liệu demo: {metadata.get('title', chunk.source_file)}\n"
        f"Nguồn: {metadata.get('source_path', '')}\n\n"
        f"{chunk.text}"
    )


def main() -> int:
    os.chdir(REPO_ROOT)
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    source_dir = args.source or Path(
        os.getenv("RESEARCH_DEMO_DIR", "documents/Research")
    )
    session_id = args.session_id or os.getenv(
        "RESEARCH_DEMO_SESSION_ID",
        DEFAULT_RESEARCH_DEMO_SESSION_ID,
    )
    session_id = str(uuid.UUID(session_id))
    upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
    session_dir = upload_dir / session_id
    report_path = Path(
        os.getenv("RESEARCH_DEMO_INDEX_REPORT", "logs/research_demo_index_report.json")
    )

    source_files = list_source_files(source_dir)
    print(f"Source:    {source_dir}")
    print(f"Session:   {session_id}")
    print(f"Documents: {len(source_files)}")
    if args.dry_run:
        for path in source_files:
            print(f"- {path.name} ({path.stat().st_size} bytes)")
        return 0

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    store = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    indexed_metadata = store.get_file_metadata(session_id)
    source_names = {path.name for path in source_files}

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "collection": store.COLLECTION_NAME,
        "session_id": session_id,
        "source_dir": str(source_dir),
        "indexed": [],
        "skipped": [],
        "failed": [],
        "pruned": [],
    }

    if not args.no_prune:
        for filename in sorted(set(indexed_metadata) - source_names):
            store.remove_file(session_id, filename)
            report["pruned"].append(filename)

    pending = []
    session_dir.mkdir(parents=True, exist_ok=True)
    for position, source_path in enumerate(source_files, start=1):
        checksum = sha256(source_path)
        previous = indexed_metadata.get(source_path.name, {})
        target_path = session_dir / source_path.name
        if not target_path.exists() or sha256(target_path) != checksum:
            shutil.copy2(source_path, target_path)
        if not args.reindex and previous.get("checksum") == checksum:
            print(f"[{position}/{len(source_files)}] SKIP {source_path.name}")
            report["skipped"].append(source_path.name)
            continue
        pending.append((position, source_path, target_path, checksum))

    parser = DocumentParser() if pending else None
    index_embedding_device = os.getenv("RESEARCH_INDEX_EMBEDDING_DEVICE", "cpu")
    embedder = Embedder(device=index_embedding_device) if pending else None

    for position, source_path, target_path, checksum in pending:
        print(f"[{position}/{len(source_files)}] INDEX {source_path.name}")
        try:
            metadata = {
                "title": source_path.stem,
                "checksum": checksum,
                "source_path": str(source_path),
                "demo_session": True,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            page_dir = session_dir / "_pages" / target_path.stem
            chunks = parser.process_file(
                target_path,
                image_output_dir=page_dir if target_path.suffix.lower() == ".pdf" else None,
                document_metadata=metadata,
            )
            if not chunks:
                raise RuntimeError("Parser returned no indexable chunks")

            embeddings = embedder.embed_documents(
                [build_embedding_text(chunk, metadata) for chunk in chunks]
            )
            store.remove_file(session_id, target_path.name)
            store.add_chunks(session_id, chunks, embeddings)
            total_chars = sum(len(chunk.text) for chunk in chunks)
            pages = len({chunk.page_number for chunk in chunks})
            report["indexed"].append(
                {
                    "filename": target_path.name,
                    "checksum": checksum,
                    "chunks": len(chunks),
                    "pages_with_content": pages,
                    "characters": total_chars,
                }
            )
            print(
                f"  -> {len(chunks)} chunks, {pages} pages, {total_chars} characters"
            )
        except Exception as exc:
            logging.exception("Failed to index %s", source_path.name)
            report["failed"].append(
                {
                    "filename": source_path.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(
        "Result: "
        f"{len(report['indexed'])} indexed, "
        f"{len(report['skipped'])} skipped, "
        f"{len(report['failed'])} failed"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
