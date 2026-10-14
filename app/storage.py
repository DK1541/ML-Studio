"""File-based storage for projects, items (labeled data) and training runs."""
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "workspace"
PROJECTS = ROOT / "projects"
PROJECTS.mkdir(parents=True, exist_ok=True)

_lock = threading.RLock()

# dataset types the app understands
DATASET_TYPES = {
    "tabular": "Table (CSV / spreadsheet rows)",
    "image_classification": "Images - classify whole image",
    "object_detection": "Images - bounding boxes",
    "segmentation": "Images - polygon masks",
    "text_classification": "Text snippets / documents",
    "handwriting": "Handwritten notes / characters (image + text label)",
}
IMAGE_TYPES = {"image_classification", "object_detection", "segmentation", "handwriting"}


def new_id(prefix=""):
    return prefix + uuid.uuid4().hex[:10]


def slug(s):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_") or "x"


def pdir(pid):
    p = PROJECTS / slug(pid)
    if not p.exists():
        raise FileNotFoundError(pid)
    return p


def _write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def list_projects():
    out = []
    for d in sorted(PROJECTS.iterdir()):
        mp = d / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            items = json.loads((d / "items.json").read_text()) if (d / "items.json").exists() else []
            m["n_items"] = len(items)
            m["n_labeled"] = sum(1 for i in items if is_labeled(m, i))
            out.append(m)
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def create_project(name, dtype, classes=None, task=None):
    if dtype not in DATASET_TYPES:
        raise ValueError("unknown dataset type")
    pid = new_id("p_")
    d = PROJECTS / pid
    (d / "files").mkdir(parents=True)
    (d / "runs").mkdir()
    meta = {
        "id": pid, "name": name or "Untitled", "type": dtype,
        "classes": [c.strip() for c in (classes or []) if c.strip()],
        "created": time.time(),
        "columns": [], "target": None, "task": task or "classification",
    }
    _write_json(d / "meta.json", meta)
    _write_json(d / "items.json", [])
    return meta


def get_meta(pid):
    return json.loads((pdir(pid) / "meta.json").read_text())


def save_meta(meta):
    with _lock:
        _write_json(pdir(meta["id"]) / "meta.json", meta)


def get_items(pid):
    return json.loads((pdir(pid) / "items.json").read_text())


def save_items(pid, items):
    with _lock:
        _write_json(pdir(pid) / "items.json", items)


def delete_project(pid):
    shutil.rmtree(pdir(pid))


def is_labeled(meta, item):
    t = meta["type"]
    if t == "object_detection":
        return bool(item.get("boxes")) or item.get("reviewed", False)
    if t == "segmentation":
        return bool(item.get("polys")) or item.get("reviewed", False)
    if t == "tabular":
        tgt = meta.get("target")
        v = (item.get("row") or {}).get(tgt) if tgt else None
        return v not in (None, "")
    return bool(item.get("label"))


def add_items(pid, new_items):
    with _lock:
        items = get_items(pid)
        items.extend(new_items)
        save_items(pid, items)
    return new_items


def update_item(pid, iid, patch):
    with _lock:
        items = get_items(pid)
        for it in items:
            if it["id"] == iid:
                it.update(patch)
                save_items(pid, items)
                return it
    raise KeyError(iid)


def delete_items(pid, ids):
    ids = set(ids)
    with _lock:
        items = get_items(pid)
        keep = []
        for it in items:
            if it["id"] in ids:
                f = it.get("file")
                if f:
                    try:
                        (pdir(pid) / "files" / f).unlink()
                    except OSError:
                        pass
            else:
                keep.append(it)
        save_items(pid, keep)


def list_runs(pid):
    out = []
    for d in sorted((pdir(pid) / "runs").iterdir()):
        rp = d / "run.json"
        if rp.exists():
            try:
                out.append(json.loads(rp.read_text()))
            except Exception:
                pass
    out.sort(key=lambda r: r.get("started", 0), reverse=True)
    return out


def run_dir(pid, rid):
    d = pdir(pid) / "runs" / slug(rid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(run):
    with _lock:
        _write_json(run_dir(run["project"], run["id"]) / "run.json", run)


def get_run(pid, rid):
    return json.loads((run_dir(pid, rid) / "run.json").read_text())
