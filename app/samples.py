"""Built-in sample datasets so new users can try the app in one click.

Everything here is generated in code (synthetic shapes, template sentences, a made-up house-price table)
or comes from datasets bundled with scikit-learn (Iris, 8x8 handwritten digits). No real photos are used.
"""
import math
import random

import cv2
import numpy as np

from . import storage as st

PREFIX = "Sample: "
COLORS = [(40, 60, 220), (50, 170, 60), (220, 120, 30), (30, 190, 220), (170, 50, 170), (200, 200, 40)]  # BGR


def _save_img(pid, arr, name, **extra):
    iid = st.new_id("i_")
    fn = iid + ".png"
    cv2.imwrite(str(st.pdir(pid) / "files" / fn), arr)
    item = {"id": iid, "file": fn, "name": name, "label": None, "boxes": [], "polys": [], "reviewed": False}
    item.update(extra)
    return item


def _canvas(rng, size=160):
    base = int(rng.integers(215, 245))
    img = np.full((size, size, 3), base, np.uint8)
    img = np.clip(img.astype(int) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    return img


def _draw_shape(img, kind, cx, cy, r, color):
    """Draw a circle / square / triangle; returns (box x,y,w,h, polygon points) in pixels."""
    if kind == "circle":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
    elif kind == "square":
        pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
    else:  # triangle
        pts = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)), pts


def _shapes_project(kind, name, classes, n, multi):
    rng = np.random.default_rng({"image_classification": 7, "object_detection": 11, "segmentation": 29}[kind])
    pid = st.create_project(PREFIX + name, kind, classes)["id"]
    items = []
    size = 160
    for k in range(n):
        img = _canvas(rng, size)
        boxes, polys = [], []
        count = int(rng.integers(1, 4)) if multi else 1
        placed = []
        for _ in range(count):
            for _try in range(20):
                r = int(rng.integers(16, 30))
                cx, cy = int(rng.integers(r + 4, size - r - 4)), int(rng.integers(r + 4, size - r - 4))
                if all(math.hypot(cx - px, cy - py) > r + pr + 6 for px, py, pr in placed):
                    break
            else:
                continue
            placed.append((cx, cy, r))
            cls = classes[(k + len(placed)) % len(classes)] if not multi else classes[int(rng.integers(len(classes)))]
            color = COLORS[int(rng.integers(len(COLORS)))]
            (x, y, w, h), pts = _draw_shape(img, cls, cx, cy, r, color)
            boxes.append({"cls": cls, "x": x / size, "y": y / size, "w": w / size, "h": h / size})
            polys.append({"cls": cls, "pts": [[px / size, py / size] for px, py in pts]})
        img = np.clip(img.astype(int) + rng.integers(-6, 6, img.shape), 0, 255).astype(np.uint8)
        extra = {}
        if kind == "image_classification":
            extra["label"] = classes[k % len(classes)]
        elif kind == "object_detection":
            extra.update(boxes=boxes, reviewed=True)
        else:
            extra.update(polys=polys, reviewed=True)
        items.append(_save_img(pid, img, f"{k}.png", **extra))
    st.add_items(pid, items)
    return pid


def _handwriting():
    from sklearn.datasets import load_digits
    d = load_digits()
    pid = st.create_project(PREFIX + "Handwritten digits", "handwriting", [str(i) for i in range(10)])["id"]
    counts, items = {}, []
    for img, lab in zip(d.images, d.target):
        if counts.get(int(lab), 0) >= 14:
            continue
        counts[int(lab)] = counts.get(int(lab), 0) + 1
        g = (255 - (img / 16.0 * 255)).astype(np.uint8)          # dark ink on white paper
        g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_CUBIC)
        items.append(_save_img(pid, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), f"digit_{lab}_{counts[int(lab)]}.png", label=str(int(lab))))
    st.add_items(pid, items)
    return pid


def _iris():
    from sklearn.datasets import load_iris
    d = load_iris()
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    meta = st.create_project(PREFIX + "Iris flowers (table)", "tabular", task="classification")
    pid = meta["id"]
    rows = [{"id": st.new_id("i_"), "row": {**{c: str(v) for c, v in zip(cols[:4], x)}, "species": d.target_names[y]}}
            for x, y in zip(d.data, d.target)]
    m = st.get_meta(pid)
    m["columns"], m["target"] = cols, "species"
    st.save_meta(m)
    st.add_items(pid, rows)
    return pid


def _houses():
    rng = np.random.default_rng(3)
    cols = ["sqft", "bedrooms", "age_years", "neighborhood", "price"]
    hood = {"Downtown": 1.35, "Lakeside": 1.2, "Suburb": 1.0, "Rural": 0.8}
    pid = st.create_project(PREFIX + "House prices (table, regression)", "tabular", task="regression")["id"]
    rows = []
    for _ in range(240):
        sqft = int(rng.integers(600, 3600))
        bed = int(np.clip(round(sqft / 900 + rng.normal(0, 0.7)), 1, 6))
        age = int(rng.integers(0, 70))
        nb = list(hood)[int(rng.integers(len(hood)))]
        price = (sqft * 140 + bed * 6000 - age * 900) * hood[nb] + rng.normal(0, 12000)
        rows.append({"id": st.new_id("i_"), "row": {"sqft": str(sqft), "bedrooms": str(bed), "age_years": str(age),
                                                    "neighborhood": nb, "price": str(int(price))}})
    m = st.get_meta(pid)
    m["columns"], m["target"] = cols, "price"
    st.save_meta(m)
    st.add_items(pid, rows)
    return pid


def _text():
    rng = random.Random(5)
    things = ["The battery", "The screen", "The delivery", "The support team", "This app", "The packaging", "The price",
              "The camera", "The sound quality", "The setup process"]
    good = ["is fantastic", "works perfectly", "exceeded my expectations", "is really impressive", "was great",
            "is excellent value", "feels premium", "is so easy to use", "made me very happy", "is wonderful"]
    bad = ["is terrible", "stopped working", "was a huge disappointment", "is really frustrating", "was awful",
           "is not worth the money", "feels cheap", "is confusing to use", "made me very angry", "is broken"]
    ok = ["is fine", "does the job", "is about average", "is okay, nothing special", "is acceptable", "works as described",
          "is neither good nor bad", "is what I expected", "is decent", "is alright"]
    pid = st.create_project(PREFIX + "Product reviews (text)", "text_classification", ["positive", "negative", "neutral"])["id"]
    items = []
    for label, pool in (("positive", good), ("negative", bad), ("neutral", ok)):
        seen = set()
        while len(seen) < 36:
            seen.add(f"{rng.choice(things)} {rng.choice(pool)}{rng.choice(['.', '!', '.'])}")
        items += [{"id": st.new_id("i_"), "text": t, "label": label} for t in sorted(seen)]
    rng.shuffle(items)
    st.add_items(pid, items)
    return pid


SAMPLES = [
    ("Iris flowers (table)", _iris),
    ("House prices (table, regression)", _houses),
    ("Product reviews (text)", _text),
    ("Handwritten digits", _handwriting),
    ("Shapes (image classes)", lambda: _shapes_project("image_classification", "Shapes (image classes)", ["circle", "square", "triangle"], 60, False)),
    ("Shapes (object detection)", lambda: _shapes_project("object_detection", "Shapes (object detection)", ["circle", "square", "triangle"], 40, True)),
    ("Shapes (segmentation)", lambda: _shapes_project("segmentation", "Shapes (segmentation)", ["circle", "square", "triangle"], 40, True)),
]


def load_samples():
    """Create the sample projects that do not exist yet. Returns the list of names created."""
    have = {p["name"] for p in st.list_projects()}
    made = []
    for name, fn in SAMPLES:
        if PREFIX + name in have:
            continue
        fn()
        made.append(name)
    return made
