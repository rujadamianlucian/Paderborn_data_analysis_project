"""Compare three Paderborn bearings side by side.

Shared controls:  channel, operating condition, repetition, spectrum type,
                  FFT window, envelope band, x-axis (Hz or shaft orders),
                  top plot (time signal or spectrogram), fault-frequency lines.
Per column:       bearing selector, top plot, spectrum.

All three columns always show the same channel, condition and repetition,
so differences between them come from the bearing, not the setup.

Run from the project root:
    python src/paderborn/viewer.py                                   # full dataset
    python src/paderborn/viewer.py data/processed/paderborn_sample.h5
"""

import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from paderborn.spectra import envelope_spectrum, fft_spectrum, spectrogram_db, welch_psd

H5_PATH = "data/processed/paderborn.h5"
CHANNELS = ["vibration_1", "phase_current_1", "phase_current_2", "force", "torque", "speed"]
WINDOWS_S = [4.0, 1.0, 0.25, 0.05]
SPECTRA = ["FFT (amplitude)", "Welch PSD (dB)", "Envelope spectrum"]
TOP_PLOTS = ["Time signal", "Spectrogram"]
DEFAULT_BEARINGS = ["K001", "KA04", "KI04"]

# Fault frequency factors for the 6203 bearing, as multiples of shaft speed.
# From nominal geometry, so approximate.
BPFO_FACTOR = 3.05
BPFI_FACTOR = 4.95
N_HARMONICS = 15

pg.setConfigOptions(antialias=False, background="w", foreground="k", imageAxisOrder="row-major")


def condition_of(measurement_name):
    """'N15_M07_F10_KA04_3' -> 'N15_M07_F10'."""
    return "_".join(measurement_name.split("_")[:3])


def shaft_hz_of(condition):
    """'N15_M07_F10' -> 25.0 Hz."""
    return int(condition[1:3]) * 100 / 60


class BearingPanel(QtWidgets.QWidget):
    """One column: bearing selector, top plot and spectrum."""

    changed = QtCore.pyqtSignal()

    def __init__(self, bearings, default):
        super().__init__()
        self.bearing_box = QtWidgets.QComboBox()
        self.bearing_box.addItems(bearings)
        if default in bearings:
            self.bearing_box.setCurrentText(default)
        self.info = QtWidgets.QLabel()

        self.top_plot = pg.PlotWidget()
        self.top_plot.setLabel("bottom", "time", units="s")
        self.top_plot.setDownsampling(auto=True, mode="peak")  # keeps 256k points smooth
        self.top_plot.setClipToView(True)

        self.spec_plot = pg.PlotWidget()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.bearing_box)
        layout.addWidget(self.info)
        layout.addWidget(self.top_plot, stretch=1)
        layout.addWidget(self.spec_plot, stretch=1)

        self.bearing_box.currentTextChanged.connect(self.changed.emit)

    def show_message(self, text):
        self.info.setText(text)
        self.top_plot.clear()
        self.spec_plot.clear()

    def update_plots(self, h5, s):
        """Redraw this column. ``s`` holds the shared settings."""
        bearing = self.bearing_box.currentText()
        meas = f"{s['condition']}_{bearing}_{s['repetition']}"
        if f"{bearing}/{meas}/{s['channel']}" not in h5:
            self.show_message(f"<b>{bearing}</b>: measurement not available")
            return

        dset = h5[f"{bearing}/{meas}/{s['channel']}"]
        x = dset[:]
        fs = float(dset.attrs["fs"])
        g = h5[bearing].attrs
        origin = g["damage_origin"]
        self.info.setText(
            f"<b>{bearing}</b> &nbsp; {g['damage_class']}"
            + (f", {origin}" if origin != "none" else "")
            + f", extent {g['extent']} &nbsp;|&nbsp; fs = {fs:.0f} Hz"
        )

        self._draw_top(x, fs, s)
        self._draw_spectrum(x, fs, s)

    def _draw_top(self, x, fs, s):
        self.top_plot.clear()
        if s["top"] == "Spectrogram":
            f, t, p = spectrogram_db(x, fs)
            keep = f <= s["fmax_hz"]
            img = pg.ImageItem(p[keep])
            img.setColorMap(pg.colormap.get("viridis"))
            img.setLevels(np.percentile(p[keep], [5, 99.5]))
            img.setRect(QtCore.QRectF(t[0], f[0], t[-1] - t[0], f[keep][-1] - f[0]))
            self.top_plot.addItem(img)
            self.top_plot.setLabel("left", "frequency", units="Hz")
            self.top_plot.setTitle(f"{s['channel']}: spectrogram (dB)")
        else:
            t = np.arange(len(x)) / fs
            self.top_plot.plot(t, x, pen=pg.mkPen("#1f77b4", width=1))
            self.top_plot.setLabel("left", s["channel"])
            self.top_plot.setTitle(s["channel"])

    def _draw_spectrum(self, x, fs, s):
        kind = s["spectrum"]
        if kind == "Welch PSD (dB)":
            f, a = welch_psd(x, fs)
            title, ylabel = "Welch PSD", "dB"
        elif kind == "Envelope spectrum":
            band = (s["band_low"], s["band_high"])
            try:
                f, a = envelope_spectrum(x, fs, band, s["window_s"])
            except ValueError as err:  # e.g. band above Nyquist for a 4 kHz channel
                self.spec_plot.clear()
                self.spec_plot.setTitle(str(err))
                return
            title, ylabel = f"Envelope spectrum, band {band[0]}–{band[1]} Hz", "amplitude"
        else:
            f, a = fft_spectrum(x, fs, s["window_s"])
            title, ylabel = f"FFT, T = {s['window_s']} s, Δf = {1 / s['window_s']:.2f} Hz", "amplitude"

        shaft = s["shaft_hz"]
        orders = s["xaxis"] == "orders"
        xvals = f / shaft if orders else f

        self.spec_plot.clear()
        self.spec_plot.plot(xvals, a, pen=pg.mkPen("#d62728", width=1))
        self.spec_plot.setTitle(title)
        self.spec_plot.setLabel("left", ylabel)
        if orders:
            self.spec_plot.setLabel("bottom", f"order (× shaft speed, {shaft:.1f} Hz)")
        else:
            self.spec_plot.setLabel("bottom", "frequency", units="Hz")

        if s["lines"]:
            for factor, color in [(BPFO_FACTOR, "#2ca02c"), (BPFI_FACTOR, "#9467bd")]:
                for k in range(1, N_HARMONICS + 1):
                    pos = k * factor if orders else k * factor * shaft
                    self.spec_plot.addItem(pg.InfiniteLine(
                        pos=pos, angle=90,
                        pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                    ))


class Viewer(QtWidgets.QWidget):
    def __init__(self, h5_path):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r")
        self.setWindowTitle(f"Paderborn bearing comparison — {h5_path}")
        bearings = sorted(self.h5.keys())

        # --- Row 1: what to look at
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.addItems(CHANNELS)

        conditions = sorted({condition_of(m) for b in self.h5 for m in self.h5[b]})
        self.condition_box = QtWidgets.QComboBox()
        self.condition_box.addItems(conditions)
        if "N15_M07_F10" in conditions:
            self.condition_box.setCurrentText("N15_M07_F10")

        self.rep_box = QtWidgets.QSpinBox()
        self.rep_box.setRange(1, 20)

        self.top_box = QtWidgets.QComboBox()
        self.top_box.addItems(TOP_PLOTS)

        self.lines_check = QtWidgets.QCheckBox("BPFO (green) / BPFI (purple)")
        self.lines_check.setChecked(True)

        # --- Row 2: how to analyse it
        self.spectrum_box = QtWidgets.QComboBox()
        self.spectrum_box.addItems(SPECTRA)

        self.window_box = QtWidgets.QComboBox()
        self.window_box.addItems([f"{w} s" for w in WINDOWS_S])

        self.band_low = QtWidgets.QSpinBox()
        self.band_high = QtWidgets.QSpinBox()
        for box, value in [(self.band_low, 2500), (self.band_high, 4000)]:
            box.setRange(10, 31_000)
            box.setSingleStep(100)
            box.setSuffix(" Hz")
            box.setValue(value)

        self.xaxis_box = QtWidgets.QComboBox()
        self.xaxis_box.addItems(["Hz", "orders"])

        self.fmax_box = QtWidgets.QComboBox()
        self.fmax_box.addItems(["1000", "5000", "32000"])

        row1 = self._row([("Channel", self.channel_box), ("Condition", self.condition_box),
                          ("Repetition", self.rep_box), ("Top plot", self.top_box)])
        row1.addWidget(self.lines_check)
        row1.addStretch()
        row2 = self._row([("Spectrum", self.spectrum_box), ("FFT window", self.window_box),
                          ("Envelope band", self.band_low), ("to", self.band_high),
                          ("x-axis", self.xaxis_box), ("Up to [Hz]", self.fmax_box)])
        row2.addStretch()

        # --- Three columns
        self.panels = [BearingPanel(bearings, b) for b in DEFAULT_BEARINGS]
        columns = QtWidgets.QHBoxLayout()
        for panel in self.panels:
            columns.addWidget(panel)
        for panel in self.panels[1:]:  # zooming one plot zooms all three
            panel.top_plot.setXLink(self.panels[0].top_plot)
            panel.spec_plot.setXLink(self.panels[0].spec_plot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(columns, stretch=1)

        # --- Any change redraws everything
        for box in (self.channel_box, self.condition_box, self.top_box, self.spectrum_box,
                    self.window_box, self.xaxis_box, self.fmax_box):
            box.currentTextChanged.connect(self.refresh)
        for box in (self.rep_box, self.band_low, self.band_high):
            box.valueChanged.connect(self.refresh)
        self.lines_check.stateChanged.connect(self.refresh)
        for panel in self.panels:
            panel.changed.connect(self.refresh)

        self.refresh()

    @staticmethod
    def _row(items):
        row = QtWidgets.QHBoxLayout()
        for label, widget in items:
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(widget)
        return row

    def settings(self):
        condition = self.condition_box.currentText()
        return {
            "channel": self.channel_box.currentText(),
            "condition": condition,
            "repetition": self.rep_box.value(),
            "top": self.top_box.currentText(),
            "spectrum": self.spectrum_box.currentText(),
            "window_s": WINDOWS_S[self.window_box.currentIndex()],
            "band_low": self.band_low.value(),
            "band_high": self.band_high.value(),
            "xaxis": self.xaxis_box.currentText(),
            "fmax_hz": int(self.fmax_box.currentText()),
            "shaft_hz": shaft_hz_of(condition),
            "lines": self.lines_check.isChecked(),
        }

    def refresh(self, *_):
        s = self.settings()
        for panel in self.panels:
            panel.update_plots(self.h5, s)

        xmax = s["fmax_hz"] / s["shaft_hz"] if s["xaxis"] == "orders" else s["fmax_hz"]
        self.panels[0].spec_plot.setXRange(0, xmax, padding=0)
        for panel in self.panels:  # dB and amplitude need different y ranges
            panel.spec_plot.enableAutoRange(axis="y")
            if s["top"] == "Spectrogram":
                panel.top_plot.setYRange(0, s["fmax_hz"], padding=0)
            else:
                panel.top_plot.enableAutoRange(axis="y")

    def closeEvent(self, event):
        self.h5.close()
        super().closeEvent(event)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else H5_PATH
    app = QtWidgets.QApplication(sys.argv)
    viewer = Viewer(path)
    viewer.resize(1700, 950)
    viewer.show()
    sys.exit(app.exec_())
