"""Convert all Paderborn .mat files in data/raw/ into one HDF5 file.

Layout:
    paderborn.h5
    └── <bearing>/                 attrs: damage_class, damage_origin, damage_method, extent
        └── <measurement>/         attrs: bearing, damage_class, speed_rpm, torque_nm,
            │                             radial_force_n, repetition, damage_origin, extent
            └── <signal>           float32 dataset, attrs: fs

Files that can't be read are skipped and listed at the end.

Run from the project root:
    python src/paderborn/to_hdf5.py
"""

import csv
import time
from pathlib import Path

import h5py

from paderborn.loader import load_measurement, parse_name


def load_bearing_table(path="metadata/bearings.csv"):
    with open(path, newline="") as fh:
        return {row["bearing"]: row for row in csv.DictReader(fh)}


def write_measurement(h5, mat_path, bearing_info):
    # Read first: if the file is broken, we fail before writing anything,
    # so no empty groups are left behind in the HDF5 file.
    signals = load_measurement(mat_path)
    info = parse_name(mat_path)
    bearing = info["bearing"]

    # Bearing group: created once, labels from the CSV
    if bearing not in h5:
        g = h5.create_group(bearing)
        for key in ("damage_class", "damage_origin", "damage_method"):
            g.attrs[key] = bearing_info[key]
        g.attrs["extent"] = int(bearing_info["extent"])

    # Measurement group: operating conditions from the file name
    m = h5[bearing].create_group(Path(mat_path).stem)
    for key, value in info.items():
        m.attrs[key] = value
    m.attrs["damage_origin"] = bearing_info["damage_origin"]
    m.attrs["extent"] = int(bearing_info["extent"])

    # One dataset per signal, at its native sampling rate
    for name, (data, fs) in signals.items():
        d = m.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)
        d.attrs["fs"] = fs


if __name__ == "__main__":
    table = load_bearing_table()
    files = sorted(Path("data/raw").glob("*/*.mat"))
    out = Path("data/processed/paderborn.h5")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(files)} files...")
    start = time.perf_counter()
    failed = []

    with h5py.File(out, "w") as h5:
        for i, path in enumerate(files, 1):
            bearing = parse_name(path)["bearing"]
            try:
                write_measurement(h5, path, table[bearing])
            except Exception as err:
                failed.append(path)
                print(f"  SKIPPED {path.name}: {err}")
            if i % 100 == 0:
                print(f"  {i}/{len(files)}  ({time.perf_counter() - start:.0f} s)")

    elapsed = time.perf_counter() - start
    print(f"Done: {len(files) - len(failed)} written, {len(failed)} skipped, {elapsed:.0f} s")
    print(f"Size: {out.stat().st_size / 1e9:.2f} GB")
    for path in failed:
        print(f"  skipped: {path}")