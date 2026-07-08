#!/usr/bin/env python3
"""Batch-index the DocJP knowledge base into Qdrant."""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.embedder import Embedder
from src.rag.parser import DocumentParser
from src.rag.vector_store import VectorStore

DOCJP_SESSION_ID = "docjp"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--file", help="Index only one manifest filename.")
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


def build_embedding_text(chunk, metadata: dict) -> str:
    """Add document identity context for semantic retrieval."""
    organization = metadata.get("organization", {})
    identity = [
        f"ナレッジベース: {metadata.get('knowledge_base', 'DocJP')}",
        f"文書: {metadata.get('title', chunk.source_file)}",
        f"分類: {metadata.get('category', '')}",
        f"会社略称: {organization.get('short_name', '')}",
        f"会社名(日本語): {organization.get('legal_name_ja', '')}",
        f"会社名(英語): {organization.get('legal_name_en', '')}",
    ]
    return "\n".join(item for item in identity if item.rsplit(": ", 1)[-1]) + (
        f"\n\n{chunk.text}"
    )


def main() -> int:
    os.chdir(REPO_ROOT)
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    source_dir = args.source or Path(
        os.getenv("DOCJP_SOURCE_DIR", "documents/Research/DocJP")
    )
    manifest_path = args.manifest or Path(
        os.getenv("DOCJP_MANIFEST_PATH", "config/docjp_manifest.json")
    )
    text_source_dir = Path(
        os.getenv("DOCJP_TEXT_SOURCE_DIR", "documents/Research/DocJP_md")
    )
    report_path = Path(
        os.getenv("DOCJP_INDEX_REPORT", "logs/docjp_index_report.json")
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", [])
    if args.file:
        documents = [item for item in documents if item["filename"] == args.file]
        if not documents:
            raise SystemExit(f"File is not present in manifest: {args.file}")

    missing = [
        item["filename"]
        for item in documents
        if not (source_dir / item["filename"]).is_file()
    ]
    if missing:
        raise SystemExit(f"Missing source documents: {missing}")

    print(f"Source:      {source_dir}")
    print(f"Text source: {text_source_dir}")
    print(f"Manifest:    {manifest_path}")
    print(f"Documents:   {len(documents)}")
    if args.dry_run:
        for item in documents:
            path = source_dir / item["filename"]
            print(f"- {item['category']:<25} {path.name} ({path.stat().st_size} bytes)")
        return 0

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    store = VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("DOCJP_COLLECTION_NAME", "docjp_knowledge"),
    )
    indexed_metadata = store.get_file_metadata(DOCJP_SESSION_ID)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "collection": store.COLLECTION_NAME,
        "indexed": [],
        "skipped": [],
        "failed": [],
        "pruned": [],
    }

    manifest_files = {item["filename"] for item in manifest.get("documents", [])}
    if not args.no_prune and not args.file:
        for filename in sorted(set(indexed_metadata) - manifest_files):
            store.remove_file(DOCJP_SESSION_ID, filename)
            report["pruned"].append(filename)

    pending = []
    for position, item in enumerate(documents, start=1):
        path = source_dir / item["filename"]
        text_source = text_source_dir / f"{path.stem}.md"
        text_source = text_source if text_source.is_file() else None
        src_checksum = sha256(path)
        md_checksum = sha256(text_source) if text_source else None
        checksum = (
            src_checksum
            if not md_checksum
            else hashlib.sha256(
                f"{src_checksum}:{md_checksum}".encode()
            ).hexdigest()
        )
        if text_source is None:
            print(
                f"[{position}/{len(documents)}] WARN no text source for "
                f"{path.name}; falling back to in-pipeline extraction."
            )
        previous = indexed_metadata.get(path.name, {})
        if not args.reindex and previous.get("checksum") == checksum:
            print(f"[{position}/{len(documents)}] SKIP {path.name}")
            report["skipped"].append(path.name)
            continue
        pending.append(
            (position, item, path, checksum, text_source, src_checksum, md_checksum)
        )

    parser = DocumentParser() if pending else None
    index_embedding_device = os.getenv("DOCJP_INDEX_EMBEDDING_DEVICE", "cpu")
    embedder = (
        Embedder(device=index_embedding_device)
        if pending
        else None
    )
    for position, item, path, checksum, text_source, src_checksum, md_checksum in pending:
        print(f"[{position}/{len(documents)}] INDEX {path.name}")
        try:
            metadata = {
                **item,
                "checksum": checksum,
                "src_checksum": src_checksum,
                "md_checksum": md_checksum,
                "text_source": text_source.name if text_source else None,
                "knowledge_base": manifest.get("knowledge_base", "DocJP"),
                "organization": manifest.get("organization", {}),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            chunks = parser.process_file(
                path,
                image_output_dir=None,
                document_metadata=metadata,
                text_source=text_source,
            )
            if not chunks:
                raise RuntimeError("Parser returned no indexable chunks")

            embeddings = embedder.embed_documents(
                [build_embedding_text(chunk, metadata) for chunk in chunks]
            )
            store.remove_file(DOCJP_SESSION_ID, path.name)
            store.add_chunks(DOCJP_SESSION_ID, chunks, embeddings)
            total_chars = sum(len(chunk.text) for chunk in chunks)
            pages = len({chunk.page_number for chunk in chunks})
            report["indexed"].append(
                {
                    "filename": path.name,
                    "checksum": checksum,
                    "text_source": text_source.name if text_source else None,
                    "chunks": len(chunks),
                    "pages_with_content": pages,
                    "characters": total_chars,
                }
            )
            print(
                f"  -> {len(chunks)} chunks, {pages} pages, {total_chars} characters"
            )
        except Exception as exc:
            logging.exception("Failed to index %s", path.name)
            report["failed"].append(
                {"filename": path.name, "error": f"{type(exc).__name__}: {exc}"}
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
