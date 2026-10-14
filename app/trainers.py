"""Training back-ends: scikit-learn (tables / text / images), PyTorch (image CNNs), Ultralytics YOLO."""
import json
import shutil
import threading
import time
import traceback

import numpy as np

from . import storage as st
from .features import ImageFeatures, classifiers, has, regressors

JOBS = {}          # run id -> live job state
CANCEL = set()     # run ids asked to stop


class Cancelled(Exception):
    pass


# =============================================================== catalog

TORCH_ARCHS = {
    "resnet18": 224, "resnet34": 224, "resnet50": 224, "resnet101": 224,
    "mobilenet_v3_small": 224, "mobilenet_v3_large": 224, "mobilenet_v2": 224,
    "efficientnet_b0": 224, "efficientnet_b1": 240, "efficientnet_v2_s": 224,
    "densenet121": 224, "convnext_tiny": 224, "regnet_y_400mf": 224,
    "shufflenet_v2_x1_0": 224, "mnasnet1_0": 224, "vit_b_16": 224, "swin_t": 224,
}
YOLO_DET = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov5nu.pt",
            "yolo11n.yaml (from scratch)", "yolov8n.yaml (from scratch)"]
YOLO_SEG = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11n-seg.yaml (from scratch)"]


def capabilities():
    return {
        "sklearn": True,
        "torch": has("torch") and has("torchvision"),
        "ultralytics": has("ultralytics"),
        "xgboost": has("xgboost"),
        "lightgbm": has("lightgbm"),
        "catboost": has("catboost"),
    }


def catalog(meta):
    t = meta["type"]
    cap = capabilities()
    out = []
    if t == "tabular":
        pool = regressors() if meta.get("task") == "regression" else classifiers()
        for n, m in pool.items():
            out.append({"id": "sk:" + n, "label": n, "group": m["group"], "params": m["params"], "available": True})
    elif t == "text_classification":
        for n, m in classifiers().items():
            out.append({"id": "sk:" + n, "label": n + " (TF-IDF)", "group": m["group"], "params": m["params"],
                        "available": True})
    elif t in ("image_classification", "handwriting"):
        for n, m in classifiers().items():
            out.append({"id": "sk:" + n, "label": n + " (HOG + pixels)", "group": "Classic: " + m["group"],
                        "params": m["params"], "available": True})
        for a in TORCH_ARCHS:
            out.append({"id": "torch:" + a, "label": a + " (PyTorch, transfer learning)", "group": "Deep learning (PyTorch)",
                        "params": {"epochs": 10, "batch": 32, "lr": 0.001, "pretrained": True, "freeze_backbone": False,
                                   "augment": t != "handwriting"},
                        "available": cap["torch"], "note": None if cap["torch"] else "pip install torch torchvision"})
    elif t in ("object_detection", "segmentation"):
        for w in (YOLO_DET if t == "object_detection" else YOLO_SEG):
            out.append({"id": "yolo:" + w, "label": w, "group": "YOLO (Ultralytics)",
                        "params": {"epochs": 50, "imgsz": 640, "batch": 8},
                        "available": cap["ultralytics"], "note": None if cap["ultralytics"] else "pip install ultralytics"})
    return out


# =============================================================== helpers

def _img_path(pid, item):
    return str(st.pdir(pid) / "files" / item["file"])


def _split(idx, y, test_size, seed=42):
    from sklearn.model_selection import train_test_split
    idx = np.asarray(idx)
    strat = None
    if y is not None:
        vals, counts = np.unique(y, return_counts=True)
        n_test = max(1, int(round(len(idx) * test_size)))
        if counts.min() >= 2 and n_test >= len(vals):
            strat = y
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def _confusion(y_true, y_pred, classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)), "recall_macro": float(np.mean(r)), "f1_macro": float(np.mean(f)),
        "classes": classes, "confusion": cm.tolist(),
        "per_class": [{"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)],
    }


def _regress_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"r2": float(r2_score(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "scatter": [[float(a), float(b)] for a, b in list(zip(y_true, y_pred))[:500]]}


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================== scikit-learn

def _tabular_frame(meta, items):
    import pandas as pd
    rows = [dict(it.get("row") or {}) for it in items]
    df = pd.DataFrame(rows)
    tgt = meta.get("target")
    if not tgt or tgt not in df.columns:
        raise ValueError("Pick a target column first (Table tab -> Target).")
    ignore = set(meta.get("ignore", []))
    df = df[df[tgt].notna() & (df[tgt].astype(str).str.strip() != "")]
    feats = [c for c in df.columns if c != tgt and c not in ignore]
    if not feats:
        raise ValueError("No feature columns left.")
    return df, feats, tgt


def _build_tabular_pre(df, feats, minmax=False):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
    num, cat = [], []
    for c in feats:
        conv = pd.to_numeric(df[c], errors="coerce")
        (num if conv.notna().mean() > 0.9 else cat).append(c)
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat:
        df[c] = df[c].astype(str)
    steps = []
    if num:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", MinMaxScaler() if minmax else StandardScaler())]), num))
    if cat:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(steps), num, cat


DENSE_ONLY = {"Gaussian Naive Bayes", "Quadratic Discriminant", "Linear Discriminant", "Hist Gradient Boosting"}


def train_sklearn(pid, meta, items, algo, params, test_size, log, metric, rdir):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    name = algo.split(":", 1)[1]
    t = meta["type"]
    regression = t == "tabular" and meta.get("task") == "regression"
    pool = regressors() if regression else classifiers()
    if name not in pool:
        raise ValueError(f"Model '{name}' is not available (library missing?)")
    spec = pool[name]
    p = dict(spec["params"])
    p.update({k: v for k, v in (params or {}).items() if k != "img_size"})
    est = spec["factory"](p)
    nonneg = name in ("Multinomial Naive Bayes", "Complement Naive Bayes")

    labeled = [i for i in items if st.is_labeled(meta, i)]
    if len(labeled) < 6:
        raise ValueError(f"Need at least 6 labeled items to train (have {len(labeled)}).")

    info = {}
    if t == "tabular":
        df, feats, tgt = _tabular_frame(meta, labeled)
        pre, num, cat = _build_tabular_pre(df, feats, minmax=nonneg)
        X = df[feats]
        info = {"features": feats, "numeric": num, "categorical": cat, "target": tgt}
        if regression:
            y = df[tgt].astype(float).values
            classes = None
        else:
            le = LabelEncoder()
            y = le.fit_transform(df[tgt].astype(str))
            classes = [str(c) for c in le.classes_]
        pipe = Pipeline([("pre", pre), ("model", est)])
    elif t == "text_classification":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = np.array([i.get("text", "") for i in labeled], dtype=object)
        le = LabelEncoder()
        y = le.fit_transform([i["label"] for i in labeled])
        classes = [str(c) for c in le.classes_]
        steps = [("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000))]
        if name in DENSE_ONLY:
            steps.append(("svd", TruncatedSVD(n_components=min(100, max(2, len(labeled) - 1)))))
        steps.append(("model", est))
        pipe = Pipeline(steps)
    else:  # image-based classification (also handwriting)
        X = np.array([_img_path(pid, i) for i in labeled], dtype=object)
        le = LabelEncoder()
        y = le.fit_transform([i["label"] for i in labeled])
        classes = [str(c) for c in le.classes_]
        steps = [("feat", ImageFeatures(size=int(params.get("img_size", 48)) if params else 48))]
        if not nonneg:
            steps.append(("sc", StandardScaler()))
        steps.append(("model", est))
        pipe = Pipeline(steps)

    if not regression and len(classes) < 2:
        raise ValueError("Need at least 2 different classes/labels.")

    tr, te = _split(np.arange(len(y)), None if regression else y, test_size)
    log(f"Training {name} on {len(tr)} samples, validating on {len(te)} ...")
    t0 = time.time()
    pipe.fit(X[tr] if not hasattr(X, "iloc") else X.iloc[tr], y[tr])
    Xte = X[te] if not hasattr(X, "iloc") else X.iloc[te]
    pred = pipe.predict(Xte)
    if regression:
        m = _regress_metrics(y[te], pred)
        log(f"R2={m['r2']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    else:
        m = _confusion(y[te], pred, classes)
        # training accuracy too, to show over/under-fitting like MATLAB's app does
        trp = pipe.predict(X[tr] if not hasattr(X, "iloc") else X.iloc[tr])
        m["train_accuracy"] = float(np.mean(trp == y[tr]))
        log(f"Validation accuracy {m['accuracy']*100:.1f}%  (train {m['train_accuracy']*100:.1f}%)")
        # misclassified examples for the scatter-style review UI
        m["wrong_ids"] = [labeled[te[i]]["id"] for i in range(len(te)) if pred[i] != y[te][i]][:200]
    m["seconds"] = round(time.time() - t0, 2)
    metric(0, {k: v for k, v in m.items() if isinstance(v, float)})
    joblib.dump({"pipeline": pipe, "classes": classes, "type": t, "regression": regression, "info": info},
                rdir / "model.joblib")
    return m


def predict_sklearn(rdir, meta, payload):
    import joblib
    import pandas as pd
    bundle = joblib.load(rdir / "model.joblib")
    pipe, classes = bundle["pipeline"], bundle["classes"]
    t = bundle["type"]
    if t == "tabular":
        feats = bundle["info"]["features"]
        row = payload["row"]
        X = pd.DataFrame([{c: row.get(c) for c in feats}])
        for c in bundle["info"]["numeric"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in bundle["info"]["categorical"]:
            X[c] = X[c].astype(str)
    elif t == "text_classification":
        X = np.array([payload["text"]], dtype=object)
    else:
        X = np.array([payload["path"]], dtype=object)
    pred = pipe.predict(X)[0]
    if bundle["regression"]:
        return {"value": float(pred)}
    out = {"label": classes[int(pred)]}
    try:
        pr = pipe.predict_proba(X)[0]
        order = np.argsort(-pr)[:5]
        out["scores"] = [{"class": classes[i], "p": float(pr[i])} for i in order]
    except Exception:
        pass
    return out


# =============================================================== PyTorch image classification

def _replace_last_linear(model, n_out):
    import torch.nn as nn
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last = name
    if last is None:
        raise ValueError("this architecture has no linear head to replace")
    parent = model
    parts = last.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    old = getattr(parent, parts[-1])
    setattr(parent, parts[-1], nn.Linear(old.in_features, n_out))
    return last


def _make_torch_model(arch, n_classes, pretrained, log):
    import torchvision
    fn = getattr(torchvision.models, arch)
    model = None
    if pretrained:
        try:
            model = fn(weights="DEFAULT")
        except Exception as e:  # offline / weights unavailable
            log(f"Could not load pretrained weights ({type(e).__name__}); training from scratch instead.")
    if model is None:
        model = fn(weights=None)
    head = _replace_last_linear(model, n_classes)
    return model, head


def _tf(size, train, augment):
    from torchvision import transforms as T
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train and augment:
        return T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
                          T.RandomRotation(10), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])


def train_torch(pid, meta, items, algo, params, test_size, log, metric, rdir, rid):
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    arch = algo.split(":", 1)[1]
    params = params or {}
    epochs, batch = int(params.get("epochs", 10)), int(params.get("batch", 32))
    lr = float(params.get("lr", 1e-3))
    size = int(params.get("img_size", TORCH_ARCHS.get(arch, 224)))
    labeled = [i for i in items if st.is_labeled(meta, i)]
    classes = sorted({i["label"] for i in labeled})
    if len(labeled) < 6 or len(classes) < 2:
        raise ValueError("Need at least 6 labeled images and 2 classes.")
    cidx = {c: k for k, c in enumerate(classes)}
    y = np.array([cidx[i["label"]] for i in labeled])
    tr, te = _split(np.arange(len(y)), y, test_size)
    dev = _device()
    log(f"Device: {dev}. {len(tr)} train / {len(te)} validation images, {len(classes)} classes.")

    class DS(Dataset):
        def __init__(self, ids, train):
            self.ids, self.train = ids, train
            self.tf = _tf(size, train, bool(params.get("augment", True)))

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, k):
            i = self.ids[k]
            img = Image.open(_img_path(pid, labeled[i])).convert("RGB")
            return self.tf(img), int(y[i])

    dl_tr = DataLoader(DS(tr, True), batch_size=batch, shuffle=True, num_workers=0)
    dl_te = DataLoader(DS(te, False), batch_size=batch, shuffle=False, num_workers=0)

    model, head = _make_torch_model(arch, len(classes), bool(params.get("pretrained", True)), log)
    if params.get("freeze_backbone"):
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(head)
    model.to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    lossf = nn.CrossEntropyLoss()
    best, best_state, best_pred = -1, None, None

    for ep in range(1, epochs + 1):
        if rid in CANCEL:
            raise Cancelled()
        model.train()
        tl, tn, tc = 0.0, 0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tn += len(yb)
            tc += (out.argmax(1) == yb).sum().item()
        sched.step()
        model.eval()
        preds, vl = [], 0.0
        with torch.no_grad():
            for xb, yb in dl_te:
                out = model(xb.to(dev))
                vl += lossf(out, yb.to(dev)).item() * len(yb)
                preds.extend(out.argmax(1).cpu().tolist())
        acc = float(np.mean(np.array(preds) == y[te]))
        metric(ep, {"train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vl / len(te), "val_acc": acc})
        log(f"epoch {ep}/{epochs}  train_loss={tl/tn:.4f} train_acc={tc/tn:.3f}  val_acc={acc:.3f}")
        if acc >= best:
            best, best_pred = acc, preds
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    m = _confusion(y[te], best_pred, classes)
    m["wrong_ids"] = [labeled[te[i]]["id"] for i in range(len(te)) if best_pred[i] != y[te][i]][:200]
    torch.save({"arch": arch, "classes": classes, "size": size, "state": best_state}, rdir / "model.pt")
    log(f"Best validation accuracy {best*100:.1f}%")
    return m


def predict_torch(rdir, payload):
    import torch
    from PIL import Image
    ck = torch.load(rdir / "model.pt", map_location="cpu", weights_only=False)
    model, _ = _make_torch_model(ck["arch"], len(ck["classes"]), False, lambda *_: None)
    model.load_state_dict(ck["state"])
    model.eval()
    x = _tf(ck["size"], False, False)(Image.open(payload["path"]).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        pr = torch.softmax(model(x), 1)[0].numpy()
    order = np.argsort(-pr)[:5]
    return {"label": ck["classes"][int(order[0])],
            "scores": [{"class": ck["classes"][i], "p": float(pr[i])} for i in order]}


# =============================================================== YOLO (detection / segmentation)

def _export_yolo(pid, meta, items, rdir, test_size, seg, log):
    labeled = [i for i in items if st.is_labeled(meta, i)]
    if len(labeled) < 4:
        raise ValueError("Need at least 4 labeled images for detection training.")
    classes = list(meta.get("classes", []))
    key = "polys" if seg else "boxes"
    for it in labeled:
        for s in it.get(key, []):
            if s["cls"] not in classes:
                classes.append(s["cls"])
    if not classes:
        raise ValueError("Define at least one class and draw some shapes.")
    cidx = {c: k for k, c in enumerate(classes)}
    tr, te = _split(np.arange(len(labeled)), None, test_size)
    base = rdir / "dataset"
    if base.exists():
        shutil.rmtree(base)
    for split, ids in (("train", tr), ("val", te)):
        (base / "images" / split).mkdir(parents=True)
        (base / "labels" / split).mkdir(parents=True)
        for i in ids:
            it = labeled[i]
            src = st.pdir(pid) / "files" / it["file"]
            dst = base / "images" / split / it["file"]
            shutil.copy(src, dst)
            lines = []
            if seg:
                for s in it.get("polys", []):
                    pts = " ".join(f"{min(max(x,0),1):.6f} {min(max(y,0),1):.6f}" for x, y in s["pts"])
                    lines.append(f"{cidx[s['cls']]} {pts}")
            else:
                for b in it.get("boxes", []):
                    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
                    lines.append(f"{cidx[b['cls']]} {cx:.6f} {cy:.6f} {b['w']:.6f} {b['h']:.6f}")
            (base / "labels" / split / (dst.stem + ".txt")).write_text("\n".join(lines))
    yaml = f"path: {base.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + \
           "".join(f"  {k}: {json.dumps(c)}\n" for k, c in enumerate(classes))
    (base / "data.yaml").write_text(yaml)
    log(f"Exported YOLO dataset: {len(tr)} train / {len(te)} val images, classes={classes}")
    return base / "data.yaml", classes


def train_yolo(pid, meta, items, algo, params, test_size, log, metric, rdir, rid):
    from ultralytics import YOLO
    params = params or {}
    seg = meta["type"] == "segmentation"
    weights = algo.split(":", 1)[1].replace(" (from scratch)", "")
    data, classes = _export_yolo(pid, meta, items, rdir, test_size, seg, log)
    model = YOLO(weights)
    hist = []

    def on_epoch(trainer):
        if trainer.epoch + 1 > trainer.epochs:   # ultralytics fires once more for the final validation
            return
        mt = {k.split("/")[-1]: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        hist.append(mt)
        metric(trainer.epoch + 1, mt)
        log(f"epoch {trainer.epoch+1}/{trainer.epochs}  " + "  ".join(f"{k}={v:.3f}" for k, v in list(mt.items())[:4]))
        if rid in CANCEL:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_epoch)
    model.train(data=str(data), epochs=int(params.get("epochs", 50)), imgsz=int(params.get("imgsz", 640)),
                batch=int(params.get("batch", 8)), project=str(rdir), name="yolo", exist_ok=True,
                verbose=False, workers=0, plots=False, patience=int(params.get("patience", 30)))
    best = rdir / "yolo" / "weights" / "best.pt"
    if not best.exists():
        best = rdir / "yolo" / "weights" / "last.pt"
    shutil.copy(best, rdir / "model.pt")
    last = hist[-1] if hist else {}
    return {"classes": classes, "final": last, "history_len": len(hist)}


def predict_yolo(rdir, meta, payload):
    from ultralytics import YOLO
    model = YOLO(str(rdir / "model.pt"))
    r = model(payload["path"], conf=float(payload.get("conf", 0.25)), verbose=False)[0]
    h, w = r.orig_shape
    names = r.names
    boxes, polys = [], []
    for i, b in enumerate(r.boxes):
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append({"cls": names[int(b.cls)], "x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h,
                      "conf": float(b.conf)})
    if r.masks is not None:
        for i, pts in enumerate(r.masks.xyn):
            polys.append({"cls": names[int(r.boxes[i].cls)], "pts": pts.tolist(), "conf": float(r.boxes[i].conf)})
    return {"boxes": boxes, "polys": polys}


# =============================================================== job runner

def start_training(pid, algo, params, test_size=0.2):
    meta = st.get_meta(pid)
    items = st.get_items(pid)
    rid = st.new_id("r_")
    rdir = st.run_dir(pid, rid)
    run = {"id": rid, "project": pid, "algo": algo, "params": params or {}, "test_size": test_size,
           "type": meta["type"], "status": "running", "started": time.time(), "log": [], "history": [],
           "metrics": None, "error": None}
    st.save_run(run)
    JOBS[rid] = run

    def log(msg):
        run["log"].append(msg)

    def metric(ep, d):
        run["history"].append({"epoch": ep, **d})

    def work():
        try:
            kind = algo.split(":", 1)[0]
            if kind == "sk":
                m = train_sklearn(pid, meta, items, algo, params, test_size, log, metric, rdir)
            elif kind == "torch":
                m = train_torch(pid, meta, items, algo, params, test_size, log, metric, rdir, rid)
            elif kind == "yolo":
                m = train_yolo(pid, meta, items, algo, params, test_size, log, metric, rdir, rid)
            else:
                raise ValueError("unknown algorithm " + algo)
            run["metrics"] = m
            run["status"] = "done"
        except Cancelled:
            run["status"] = "cancelled"
            log("Cancelled.")
        except Exception as e:
            run["status"] = "error"
            run["error"] = f"{type(e).__name__}: {e}"
            log(traceback.format_exc())
        finally:
            run["finished"] = time.time()
            CANCEL.discard(rid)
            st.save_run(run)

    threading.Thread(target=work, daemon=True).start()
    return run


def predict(pid, rid, payload):
    run = st.get_run(pid, rid)
    meta = st.get_meta(pid)
    rdir = st.run_dir(pid, rid)
    kind = run["algo"].split(":", 1)[0]
    if kind == "sk":
        return predict_sklearn(rdir, meta, payload)
    if kind == "torch":
        return predict_torch(rdir, payload)
    return predict_yolo(rdir, meta, payload)
