from pathlib import Path
import numpy as np
from scipy.io import loadmat

CLASS_FROM_PREFIX = {
    "K0": "healthy",
    "KA": "outer",     # outer ring damage
    "KI": "inner",     # inner ring damage
    "KB": "combined",  # both
}

def text(x):
    return x[0] if x.size > 0 else ""


def load_measurement(path):
    """Read one Paderborn .mat file -> dict of {signal_name: (data, fs)}."""
    path = Path(path)
    mat = loadmat(path)
    root = mat[path.stem]          # the struct is named after the file
    X = root["X"][0, 0]
    Y = root["Y"][0, 0]

    signals = {}
    for i in range(Y.shape[1]):
        sig = Y[0, i]
        name = str(text(sig["Name"]))
        data = sig["Data"].ravel().astype(np.float32)

        x_idx = int(sig["XIndex"].ravel()[0]) - 1   # MATLAB 1-based -> Python 0-based
        t = X[0, x_idx]["Data"].ravel()
        fs = (t.size - 1) / (t[-1] - t[0])

        signals[name] = (data, fs)
    return signals


def parse_name(path):
    """'N09_M07_F10_K001_1' -> operating conditions and labels."""
    stem = Path(path).stem
    n, m, f, bearing, rep = stem.split("_")
    return {
        "bearing": bearing,
        "damage_class": CLASS_FROM_PREFIX[bearing[:2]],
        "speed_rpm": int(n[1:]) * 100,      # N09 -> 900
        "torque_nm": int(m[1:]) / 10,       # M07 -> 0.7
        "radial_force_n": int(f[1:]) * 100, # F10 -> 1000
        "repetition": int(rep),
    }

if __name__ == "__main__":
    signals = load_measurement("data/raw/K001/N09_M07_F10_K001_1.mat")
    for name, (data, fs) in signals.items():
        print(f"{name:<25} {data.size:>8} samples  fs = {fs:8.1f} Hz")

    print(parse_name("data/raw/K001/N09_M07_F10_K001_1.mat"))
    print(parse_name("N15_M01_F10_KA04_7"))