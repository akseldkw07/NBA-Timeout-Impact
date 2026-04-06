"""Disk I/O for saved queries and conversations."""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import polars as pl

from nba_timeout_impact.constants import NBAConstants

BASE_DIR = Path(__file__).parent
SAVED_QUERIES_DIR = BASE_DIR / "saved_queries"
CONVERSATIONS_DIR = BASE_DIR / "conversations"
DATASETS_DIR = NBAConstants.NBA_DATA_DIR


def _ensure_dirs():
    SAVED_QUERIES_DIR.mkdir(exist_ok=True)
    CONVERSATIONS_DIR.mkdir(exist_ok=True)


def generate_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _read_index(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _write_index(path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# --- Saved Queries ---


def load_saved_queries_index():
    return _read_index(SAVED_QUERIES_DIR / "index.json")


def save_query(title, description, sql, explanation, df, fig=None, metadata=None):
    _ensure_dirs()
    qid = generate_id()
    qdir = SAVED_QUERIES_DIR / qid
    qdir.mkdir(parents=True)

    # Write data
    df.write_parquet(str(qdir / "data.parquet"), compression="zstd", compression_level=3)

    fig_json = None
    if fig is not None:
        fig_json = fig.to_json()
        (qdir / "figure.json").write_text(fig_json)

    meta = {
        "id": qid,
        "title": title,
        "description": description,
        "sql": sql,
        "explanation": explanation,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "pinned": False,
        "row_count": df.height,
        "has_figure": fig is not None,
        "tables_used": (metadata or {}).get("tables_used", []),
        "columns_used": (metadata or {}).get("columns_used", {}),
    }
    (qdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Update index
    index = load_saved_queries_index()
    index_entry = {k: v for k, v in meta.items() if k != "explanation"}
    index_entry["size_bytes"] = sum(f.stat().st_size for f in qdir.iterdir())
    index.append(index_entry)
    _write_index(SAVED_QUERIES_DIR / "index.json", index)
    return qid


def load_saved_query(query_id):
    qdir = SAVED_QUERIES_DIR / query_id
    meta = json.loads((qdir / "meta.json").read_text())
    df = pl.read_parquet(qdir / "data.parquet")
    fig = None
    fig_path = qdir / "figure.json"
    if fig_path.exists():
        import plotly.io as pio

        fig = pio.from_json(fig_path.read_text())
    return {**meta, "df": df, "fig": fig}


def delete_saved_query(query_id):
    qdir = SAVED_QUERIES_DIR / query_id
    if qdir.exists():
        shutil.rmtree(qdir)
    index = [e for e in load_saved_queries_index() if e["id"] != query_id]
    _write_index(SAVED_QUERIES_DIR / "index.json", index)


def update_saved_query(query_id, **updates):
    qdir = SAVED_QUERIES_DIR / query_id
    meta_path = qdir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    meta.update(updates)
    meta["updated_at"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # Update index entry
    index = load_saved_queries_index()
    for entry in index:
        if entry["id"] == query_id:
            entry.update({k: v for k, v in updates.items() if k in entry})
            entry["updated_at"] = meta["updated_at"]
            break
    _write_index(SAVED_QUERIES_DIR / "index.json", index)


def get_pinned_queries():
    return [e for e in load_saved_queries_index() if e.get("pinned")]


def get_recent_queries(n=5):
    index = load_saved_queries_index()
    return sorted(index, key=lambda e: e.get("updated_at", ""), reverse=True)[:n]


# --- Conversations ---


def save_conversation(conv_id, messages, results):
    _ensure_dirs()
    # Strip non-serializable objects from results
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items() if k not in ("df", "fig")}
        if r.get("df") is not None:
            cr["row_count"] = r["df"].height
        clean_results.append(cr)

    data = {
        "id": conv_id,
        "started_at": messages[0]["content"][:100] if messages else "",
        "last_activity": datetime.now().isoformat(),
        "messages": messages,
        "results": clean_results,
    }
    conv_path = CONVERSATIONS_DIR / f"{conv_id}.json"
    conv_path.write_text(json.dumps(data, indent=2, default=str))

    # Update index
    error_count = sum(1 for r in clean_results if r.get("error"))
    query_count = len(clean_results)
    first_q = ""
    for m in messages:
        if m["role"] == "user":
            first_q = m["content"][:100]
            break

    index = load_conversations_index()
    # Update existing or append
    entry = {
        "id": conv_id,
        "started_at": data.get("started_at", ""),
        "last_activity": data["last_activity"],
        "message_count": len(messages),
        "query_count": query_count,
        "error_count": error_count,
        "size_bytes": conv_path.stat().st_size,
        "first_question": first_q,
    }
    index = [e for e in index if e["id"] != conv_id]
    index.append(entry)
    _write_index(CONVERSATIONS_DIR / "index.json", index)


def load_conversation(conv_id):
    conv_path = CONVERSATIONS_DIR / f"{conv_id}.json"
    return json.loads(conv_path.read_text())


def load_conversations_index():
    return _read_index(CONVERSATIONS_DIR / "index.json")


def delete_conversation(conv_id):
    conv_path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if conv_path.exists():
        conv_path.unlink()
    index = [e for e in load_conversations_index() if e["id"] != conv_id]
    _write_index(CONVERSATIONS_DIR / "index.json", index)


# --- Storage sizes ---


def _dir_size(path):
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def get_storage_sizes():
    return {
        "datasets": _dir_size(DATASETS_DIR),
        "saved_queries": _dir_size(SAVED_QUERIES_DIR),
        "conversations": _dir_size(CONVERSATIONS_DIR),
    }


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"
