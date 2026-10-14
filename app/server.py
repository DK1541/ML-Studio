"""FastAPI server: projects, datasets, labeling, training, prediction, export."""
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import storage as st
from . import trainers as tr

app = FastAPI(title="ML Studio")
STATIC = Path(__file__).parent / "static"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@app.exception_handler(FileNotFoundError)
async def nf(_, e):
    return JSONResponse({"detail": "not found"}, status_code=404)


@app.exception_handler(ValueError)
async def ve(_, e):
    return JSONResponse({"detail": str(e)}, status_code=400)


# ------------------------------------------------------------------ models
class NewProject(BaseModel):
    name: str
    type: str
    classes: List[str] = []
    task: Optional[str] = None


class ClassesBody(BaseModel):
    classes: List[str]


class ItemPatch(BaseModel):
    data: Dict[str, Any]


class NewItem(BaseModel):
    text: Optional[str] = None
    label: Optional[str] = None
    row: Optional[Dict[str, Any]] = None


class TabularCfg(BaseModel):
    target: Optional[str] = None
    task: Optional[str] = None
    ignore: Optional[List[str]] = None
    columns: Optional[List[str]] = None


class TrainBody(BaseModel):
    algo: str
    params: Dict[str, Any] = {}
    test_size: float = 0.2


class PredictBody(BaseModel):
    run: str
    row: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    item: Optional[str] = None
    conf: float = 0.25


class DeleteBody(BaseModel):
    ids: List[str]


# ------------------------------------------------------------------ meta
@app.get("/api/capabilities")
def capabilities():
    return {"libs": tr.capabilities(), "types": st.DATASET_TYPES}


@app.get("/api/projects")
def projects():
    return st.list_projects()


@app.post("/api/projects")
def create(b: NewProject):
    return st.create_project(b.name, b.type, b.classes, b.task)


@app.delete("/api/projects/{pid}")
def delete(pid: str):
    st.delete_project(pid)
    return {"ok": True}


@app.get("/api/projects/{pid}")
def project(pid: str):
    meta = st.get_meta(pid)
    items = st.get_items(pid)
    return {"meta": meta, "items": items}


@app.post("/api/projects/{pid}/classes")
def set_classes(pid: str, b: ClassesBody):
    meta = st.get_meta(pid)
    meta["classes"] = [c.strip() for c in b.classes if c.strip()]
    st.save_meta(meta)
    return meta


@app.post("/api/projects/{pid}/tabular")
def tabular_cfg(pid: str, b: TabularCfg):
    meta = st.get_meta(pid)
    for k in ("target", "task", "ignore", "columns"):
        v = getattr(b, k)
        if v is not None:
            meta[k] = v
    st.save_meta(meta)
    return meta


# ------------------------------------------------------------------ items
def _rows_from_csv(raw: bytes):
    txt = raw.decode("utf-8-sig", errors="replace")
    rd = csv.DictReader(io.StringIO(txt))
    return rd.fieldnames or [], list(rd)


@app.post("/api/projects/{pid}/upload")
async def upload(pid: str, files: List[UploadFile] = File(...)):
    meta = st.get_meta(pid)
    t = meta["type"]
    new, errors = [], []
    fdir = st.pdir(pid) / "files"
    for f in files:
        name = f.filename or "file"
        ext = Path(name).suffix.lower()
        raw = await f.read()
        if t in st.IMAGE_TYPES:
            if ext == ".zip":
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for zi in z.infolist():
                        zext = Path(zi.filename).suffix.lower()
                        if zi.is_dir() or zext not in IMG_EXT or "__MACOSX" in zi.filename:
                            continue
                        parts = Path(zi.filename).parts
                        label = parts[-2] if len(parts) > 1 and t == "image_classification" else None
                        iid = st.new_id("i_")
                        fn = iid + zext
                        (fdir / fn).write_bytes(z.read(zi))
                        new.append(_img_item(iid, fn, Path(zi.filename).name, label))
            elif ext in IMG_EXT:
                iid = st.new_id("i_")
                fn = iid + ext
                (fdir / fn).write_bytes(raw)
                new.append(_img_item(iid, fn, name, None))
            else:
                errors.append(f"{name}: not an image")
        elif t == "tabular":
            if ext not in (".csv", ".tsv", ".txt"):
                errors.append(f"{name}: upload a .csv file")
                continue
            if ext == ".tsv":
                raw = raw.replace(b"\t", b",")
            cols, rows = _rows_from_csv(raw)
            for c in cols:
                if c not in meta["columns"]:
                    meta["columns"].append(c)
            for r in rows:
                new.append({"id": st.new_id("i_"), "row": {k: v for k, v in r.items() if k is not None}})
            if not meta.get("target") and cols:
                meta["target"] = cols[-1]
        elif t == "text_classification":
            txt = raw.decode("utf-8", errors="replace")
            if ext == ".csv":
                cols, rows = _rows_from_csv(raw)
                tcol = next((c for c in cols if c.lower() in ("text", "sentence", "content", "message")), cols[0])
                lcol = next((c for c in cols if c.lower() in ("label", "class", "category", "target")), None)
                for r in rows:
                    new.append({"id": st.new_id("i_"), "text": r.get(tcol, ""), "label": (r.get(lcol) if lcol else None) or None})
            else:
                label = None
                for line in txt.splitlines():
                    if line.strip():
                        new.append({"id": st.new_id("i_"), "text": line.strip(), "label": label})
    # keep classes in sync with imported labels
    labels = {i.get("label") for i in new if i.get("label")}
    for lb in sorted(labels):
        if lb not in meta["classes"]:
            meta["classes"].append(lb)
    st.save_meta(meta)
    st.add_items(pid, new)
    return {"added": len(new), "errors": errors}


def _img_item(iid, fn, name, label):
    return {"id": iid, "file": fn, "name": name, "label": label, "boxes": [], "polys": [], "reviewed": False}


@app.post("/api/projects/{pid}/items")
def add_item(pid: str, b: NewItem):
    meta = st.get_meta(pid)
    if meta["type"] == "tabular":
        row = b.row or {}
        for c in row:
            if c not in meta["columns"]:
                meta["columns"].append(c)
        st.save_meta(meta)
        item = {"id": st.new_id("i_"), "row": row}
    elif meta["type"] == "text_classification":
        item = {"id": st.new_id("i_"), "text": b.text or "", "label": b.label}
        if b.label and b.label not in meta["classes"]:
            meta["classes"].append(b.label)
            st.save_meta(meta)
    else:
        raise ValueError("use /upload for image datasets")
    st.add_items(pid, [item])
    return item


@app.patch("/api/projects/{pid}/items/{iid}")
def patch_item(pid: str, iid: str, b: ItemPatch):
    meta = st.get_meta(pid)
    data = dict(b.data)
    # auto-register new classes
    cand = []
    if data.get("label"):
        cand.append(data["label"])
    for s in (data.get("boxes") or []) + (data.get("polys") or []):
        cand.append(s["cls"])
    changed = False
    for c in cand:
        if c and c not in meta["classes"]:
            meta["classes"].append(c)
            changed = True
    if "row" in data:
        for c in data["row"]:
            if c not in meta["columns"]:
                meta["columns"].append(c)
                changed = True
    if changed:
        st.save_meta(meta)
    return st.update_item(pid, iid, data)


@app.post("/api/projects/{pid}/delete-items")
def delete_items(pid: str, b: DeleteBody):
    st.delete_items(pid, b.ids)
    return {"ok": True}


@app.get("/api/projects/{pid}/file/{fn}")
def get_file(pid: str, fn: str):
    p = st.pdir(pid) / "files" / Path(fn).name
    if not p.exists():
        raise FileNotFoundError(fn)
    return FileResponse(p)


# ------------------------------------------------------------------ training
@app.get("/api/projects/{pid}/catalog")
def catalog(pid: str):
    return tr.catalog(st.get_meta(pid))


@app.post("/api/projects/{pid}/train")
def train(pid: str, b: TrainBody):
    return tr.start_training(pid, b.algo, b.params, b.test_size)


@app.get("/api/projects/{pid}/runs")
def runs(pid: str):
    out = []
    for r in st.list_runs(pid):
        live = tr.JOBS.get(r["id"])
        out.append(live if live else r)
    return out


@app.get("/api/projects/{pid}/runs/{rid}")
def run(pid: str, rid: str):
    return tr.JOBS.get(rid) or st.get_run(pid, rid)


@app.post("/api/projects/{pid}/runs/{rid}/cancel")
def cancel(pid: str, rid: str):
    tr.CANCEL.add(rid)
    return {"ok": True}


@app.delete("/api/projects/{pid}/runs/{rid}")
def delete_run(pid: str, rid: str):
    shutil.rmtree(st.run_dir(pid, rid), ignore_errors=True)
    tr.JOBS.pop(rid, None)
    return {"ok": True}


@app.get("/api/projects/{pid}/runs/{rid}/model")
def download_model(pid: str, rid: str):
    d = st.run_dir(pid, rid)
    for n in ("model.pt", "model.joblib"):
        if (d / n).exists():
            return FileResponse(d / n, filename=f"{pid}_{rid}_{n}")
    raise FileNotFoundError(rid)


@app.post("/api/projects/{pid}/predict")
async def predict(pid: str, b: PredictBody):
    meta = st.get_meta(pid)
    payload: Dict[str, Any] = {"conf": b.conf}
    if meta["type"] == "tabular":
        payload["row"] = b.row or {}
    elif meta["type"] == "text_classification":
        payload["text"] = b.text or ""
    else:
        items = {i["id"]: i for i in st.get_items(pid)}
        if not b.item or b.item not in items:
            raise ValueError("item required")
        payload["path"] = str(st.pdir(pid) / "files" / items[b.item]["file"])
    return tr.predict(pid, b.run, payload)


@app.post("/api/projects/{pid}/predict-upload/{rid}")
async def predict_upload(pid: str, rid: str, file: UploadFile = File(...), conf: float = 0.25):
    """Run a trained image model on a fresh image that is not part of the dataset."""
    tmp = st.run_dir(pid, rid) / ("_tmp" + Path(file.filename or "x.png").suffix)
    tmp.write_bytes(await file.read())
    try:
        return tr.predict(pid, rid, {"path": str(tmp), "conf": conf})
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ export
@app.get("/api/projects/{pid}/export")
def export(pid: str):
    meta = st.get_meta(pid)
    items = st.get_items(pid)
    t = meta["type"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps({"meta": meta, "items": items}, indent=1))
        if t == "tabular":
            cols = meta["columns"]
            sio = io.StringIO()
            w = csv.DictWriter(sio, fieldnames=cols)
            w.writeheader()
            for i in items:
                w.writerow({c: (i.get("row") or {}).get(c, "") for c in cols})
            z.writestr("data.csv", sio.getvalue())
        elif t == "text_classification":
            sio = io.StringIO()
            w = csv.writer(sio)
            w.writerow(["text", "label"])
            for i in items:
                w.writerow([i.get("text", ""), i.get("label") or ""])
            z.writestr("data.csv", sio.getvalue())
        else:
            classes = list(meta["classes"])
            for i in items:
                src = st.pdir(pid) / "files" / i["file"]
                if t in ("image_classification", "handwriting"):
                    z.write(src, f"{(i.get('label') or '_unlabeled')}/{i['file']}")
                else:
                    z.write(src, f"images/{i['file']}")
                    lines = []
                    for s in i.get("boxes", []):
                        if s["cls"] not in classes:
                            classes.append(s["cls"])
                        lines.append(f"{classes.index(s['cls'])} {s['x']+s['w']/2:.6f} {s['y']+s['h']/2:.6f} {s['w']:.6f} {s['h']:.6f}")
                    for s in i.get("polys", []):
                        if s["cls"] not in classes:
                            classes.append(s["cls"])
                        lines.append(f"{classes.index(s['cls'])} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in s["pts"]))
                    z.writestr(f"labels/{Path(i['file']).stem}.txt", "\n".join(lines))
            if t in ("object_detection", "segmentation"):
                z.writestr("classes.txt", "\n".join(classes))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{meta["name"]}.zip"'})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
