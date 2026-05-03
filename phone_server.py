import argparse
import datetime
import ipaddress
import os
import random
import socket
import ssl
import tempfile
import threading

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

IMG_SIZE   = 32
HIDDEN     = 64
ROUNDS     = 10
LR         = 0.005
LOC_EPOCHS = 3     
BATCH      = 8
DP_CLIP    = 1.0
DP_NOISE   = 0.02
MAX_PPL    = 20     
MAX_IMGS   = 25
SUPP_PER_CLASS = 4  

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

G = {
    "round":          0,
    "global_weights": None,
    "phone_update":   None,
    "num_classes":    0,
    "class_names":    [],
    "history":        [],
    "status":         "waiting for phone",
    "lock":           threading.Lock(),
}
DATA = {
    "all_X": None, "all_y": None,  
    "laptop_X": None, "laptop_y": None,
    "test_X":   None, "test_y":   None,
}

class TinyMLP:
    def __init__(self, inp, hid, out):
        self.W1 = (np.random.randn(inp, hid) * np.sqrt(2.0/inp)).astype(np.float32)
        self.b1 = np.zeros(hid,  dtype=np.float32)
        self.W2 = (np.random.randn(hid, out) * np.sqrt(2.0/hid)).astype(np.float32)
        self.b2 = np.zeros(out,  dtype=np.float32)
        self.X = self.z1 = self.a1 = self.p = None

    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)
        z2      = self.a1 @ self.W2 + self.b2
        z2     -= z2.max(1, keepdims=True)
        e = np.exp(z2)
        self.p = e / e.sum(1, keepdims=True)
        return self.p

    def backward(self, y):
        n = len(y)
        dz2 = self.p.copy(); dz2[range(n), y] -= 1.0; dz2 /= n
        self.dW2 = self.a1.T @ dz2; self.db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (self.z1 > 0)
        self.dW1 = self.X.T @ dz1; self.db1 = dz1.sum(0)

    def dp_step(self):
        for g in [self.dW1, self.db1, self.dW2, self.db2]:
            norm = float(np.linalg.norm(g))
            if norm > DP_CLIP: g *= DP_CLIP / (norm + 1e-8)
            g += (np.random.randn(*g.shape) * DP_NOISE).astype(np.float32)

    def step(self):
        self.W1 -= LR*self.dW1; self.b1 -= LR*self.db1
        self.W2 -= LR*self.dW2; self.b2 -= LR*self.db2

    def accuracy(self, X, y, bs=64):
        correct = 0
        for i in range(0, len(X), bs):
            correct += int((self.forward(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum())
        return 100.0 * correct / max(len(y), 1)

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("W1","b1","W2","b2")}

    def from_dict(self, d):
        for k in ("W1","b1","W2","b2"):
            setattr(self, k, np.array(d[k], dtype=np.float32).reshape(getattr(self, k).shape))


def fedavg(updates):
    total  = sum(n for _, n in updates)
    shapes = {k: np.array(G["global_weights"][k], dtype=np.float32).shape
              for k in ("W1","b1","W2","b2")}
    out    = {}
    for k in ("W1","b1","W2","b2"):
        out[k] = sum(
            np.array(w[k], dtype=np.float32).reshape(shapes[k]) * (n / total)
            for w, n in updates
        ).tolist()
    return out


def laptop_train():
    X, y  = DATA["laptop_X"], DATA["laptop_y"]
    m     = TinyMLP(X.shape[1], HIDDEN, G["num_classes"])
    m.from_dict(G["global_weights"])
    perm  = np.random.permutation(len(X))
    X, y  = X[perm], y[perm]
    for _ in range(LOC_EPOCHS):
        for i in range(0, len(X) - BATCH + 1, BATCH):
            xb, yb = X[i:i+BATCH], y[i:i+BATCH]
            m.forward(xb); m.backward(yb); m.dp_step(); m.step()
    return m.to_dict(), len(X)


def run_round():
    rnd         = G["round"] + 1
    G["status"] = f"round {rnd} — laptop training"
    print(f"\n[Round {rnd}/{ROUNDS}] Laptop local training …", flush=True)

    lw, ln  = laptop_train()
    updates = [(lw, ln)]
    if G["phone_update"] is not None:
        updates.append(G["phone_update"])
        note = f"laptop + phone ({G['phone_update'][1]} samples)"
    else:
        note = "laptop only"

    G["status"]         = "aggregating"
    G["global_weights"] = fedavg(updates)

    ev = TinyMLP(DATA["test_X"].shape[1], HIDDEN, G["num_classes"])
    ev.from_dict(G["global_weights"])
    acc = ev.accuracy(DATA["test_X"], DATA["test_y"])

    G["history"].append({"round": rnd, "acc": round(float(acc), 2),
                         "clients": len(updates), "note": note})
    G["round"]        = rnd
    G["phone_update"] = None
    G["status"]       = "complete" if rnd >= ROUNDS else "waiting for phone"
    print(f"  acc={acc:.1f}%  [{note}]", flush=True)


def _read_img(path):
    try:
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.array(img, dtype=np.float32).flatten() / 255.0
    except Exception:
        return np.zeros(IMG_SIZE * IMG_SIZE * 3, dtype=np.float32)


def _finalise(X, y, names):
    perm  = np.random.permutation(len(X))
    X, y  = X[perm], y[perm]
    split = max(len(X) // 5, len(names))
    DATA["all_X"],    DATA["all_y"]    = X, y     
    DATA["test_X"],   DATA["test_y"]   = X[:split],  y[:split]
    DATA["laptop_X"], DATA["laptop_y"] = X[split:],  y[split:]
    G["num_classes"]    = len(names)
    G["class_names"]    = list(names)
    G["global_weights"] = TinyMLP(X.shape[1], HIDDEN, len(names)).to_dict()
    print(f"  {len(names)} identities  "
          f"laptop={len(DATA['laptop_X'])}  test={len(DATA['test_X'])}", flush=True)


def prepare_dataset(path):
    print(f"Scanning {path} …", flush=True)
    people = []
    for name in sorted(os.listdir(path)):
        d    = os.path.join(path, name)
        if not os.path.isdir(d): continue
        imgs = [os.path.join(d, f) for f in os.listdir(d)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        if len(imgs) >= 5:      # lowered from 8
            people.append((name, imgs))
    people.sort(key=lambda x: -len(x[1]))
    people = people[:MAX_PPL]
    if not people:
        raise FileNotFoundError(
            f"No people with ≥5 images in '{path}'.\n"
            "Expected: dataset/PersonName/photo.jpg"
        )
    print(f"Loading {len(people)} identities …", flush=True)
    X, y = [], []
    for label, (name, imgs) in enumerate(people):
        random.shuffle(imgs)
        for p in imgs[:MAX_IMGS]:
            X.append(_read_img(p)); y.append(label)
    _finalise(np.array(X, np.float32), np.array(y, np.int32),
              [p[0] for p in people])


def prepare_demo(nc=5):
    print("Demo mode — synthetic data …", flush=True)
    rng = np.random.default_rng(42)
    pat = rng.random((nc, IMG_SIZE*IMG_SIZE*3)).astype(np.float32)
    X, y = [], []
    for c in range(nc):
        for _ in range(20):
            X.append(np.clip(pat[c]+rng.normal(0,.25,pat[c].shape).astype(np.float32),0,1))
            y.append(c)
    _finalise(np.array(X,np.float32), np.array(y,np.int32),
              [f"Person_{i+1:02d}" for i in range(nc)])



@app.route("/")
def index():
    return render_template("phone.html")


@app.route("/api/info")
def api_info():
    return jsonify({
        "class_names":  G["class_names"],
        "num_classes":  G["num_classes"],
        "img_size":     IMG_SIZE,
        "hidden":       HIDDEN,
        "total_rounds": ROUNDS,
    })


@app.route("/api/weights")
def api_get_weights():
    return jsonify({
        "weights":      G["global_weights"],
        "round":        G["round"],
        "total_rounds": ROUNDS,
        "num_classes":  G["num_classes"],
    })


@app.route("/api/weights", methods=["POST"])
def api_post_weights():
    body = request.get_json(force=True) or {}
    with G["lock"]:
        G["phone_update"] = (body["weights"], int(body.get("num_samples", 1)))
        G["status"]       = "phone weights received"
    threading.Thread(target=run_round, daemon=True).start()
    return jsonify({"status": "ok", "next_round": G["round"] + 1})


@app.route("/api/status")
def api_status():
    return jsonify({
        "round":        G["round"],
        "total_rounds": ROUNDS,
        "status":       G["status"],
        "history":      G["history"],
    })


@app.route("/api/supplement")
def api_supplement():
    X, y = DATA["all_X"], DATA["all_y"]
    nc   = G["num_classes"]
    sup_x, sup_y = [], []

    for cls in range(nc):
        idxs = np.where(y == cls)[0]
        n    = min(SUPP_PER_CLASS, len(idxs))
        chosen = np.random.choice(idxs, n, replace=False)
        for i in chosen:
            sup_x.append((X[i] * 255).astype(np.uint8).tolist())  # uint8 for bandwidth
            sup_y.append(int(cls))

    return jsonify({
        "images":      sup_x,
        "labels":      sup_y,
        "num_classes": nc,
        "num_samples": len(sup_x),
    })


@app.route("/api/add_person", methods=["POST"])
def api_add_person():
    body = request.get_json(force=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if name in G["class_names"]:
        return jsonify({"error": f"'{name}' already exists",
                        "class_names": G["class_names"],
                        "num_classes": G["num_classes"]}), 409

    with G["lock"]:
        G["class_names"].append(name)
        old_nc = G["num_classes"]
        new_nc = old_nc + 1
        G["num_classes"] = new_nc

        old_W2 = np.array(G["global_weights"]["W2"], dtype=np.float32).reshape(HIDDEN, old_nc)
        new_col = (np.random.randn(HIDDEN, 1) * np.sqrt(2.0/HIDDEN)).astype(np.float32)
        new_W2  = np.concatenate([old_W2, new_col], axis=1)

        old_b2 = np.array(G["global_weights"]["b2"], dtype=np.float32)
        new_b2 = np.append(old_b2, 0.0).astype(np.float32)

        G["global_weights"]["W2"] = new_W2.tolist()
        G["global_weights"]["b2"] = new_b2.tolist()

        print(f"  Added person '{name}' — now {new_nc} identities", flush=True)

    return jsonify({
        "status":      "ok",
        "class_names": G["class_names"],
        "num_classes": G["num_classes"],
        "new_index":   new_nc - 1,
    })


BLAZEFACE_DIR = os.path.join(os.path.dirname(__file__), "static", "blazeface")

@app.route("/static/blazeface/<path:filename>")
def serve_blazeface(filename):
    return send_from_directory(BLAZEFACE_DIR, filename)


def generate_cert(ip_str):
    key  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FL Local Server")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subj).issuer_name(subj)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.IPv4Address(ip_str)),
                x509.DNSName("localhost"),
            ]), critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cf = tempfile.NamedTemporaryFile(suffix=".crt", delete=False)
    kf = tempfile.NamedTemporaryFile(suffix=".key", delete=False)
    cf.write(cert.public_bytes(serialization.Encoding.PEM)); cf.close()
    kf.write(key.private_bytes(serialization.Encoding.PEM,
             serialization.PrivateFormat.TraditionalOpenSSL,
             serialization.NoEncryption())); kf.close()
    return cf.name, kf.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./dataset")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42)

    if args.demo:
        prepare_demo()
    else:
        prepare_dataset(args.data_path)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except Exception:
        ip = "127.0.0.1"

    bf_ready = os.path.isfile(os.path.join(BLAZEFACE_DIR, "model.json"))
    if not bf_ready:
        print("\n  ⚠  BlazeFace not downloaded yet.")
        print("     Run:  python setup.py  (with internet) to enable offline detection.\n")

    ssl_ctx = None; cert_path = key_path = None
    if HAS_CRYPTO:
        try:
            cert_path, key_path = generate_cert(ip)
            ssl_ctx = (cert_path, key_path)
            proto = "https"
        except Exception as e:
            print(f"TLS setup failed ({e}) — using HTTP (camera will not work)")
            proto = "http"
    else:
        proto = "http"
        print("Install 'cryptography' for HTTPS:  pip install cryptography")

    print(f"\n{'='*58}")
    print(f"  Server   →  {proto}://{ip}:{args.port}")
    print(f"  Phone:       open that URL in Chrome")
    if ssl_ctx:
        print(f"  Chrome:      tap Advanced → Proceed (self-signed cert)")
    print(f"  BlazeFace:   {'offline ✓' if bf_ready else 'needs python setup.py first'}")
    print(f"{'='*58}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False,
            threaded=True, ssl_context=ssl_ctx)
