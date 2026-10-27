"""Check the spectral tools on a synthetic bearing-like signal.

The test signal mimics an outer-ring fault: short impacts repeating at
76 Hz that ring a resonance at 3200 Hz, buried in noise. A plain FFT shows
energy around 3200 Hz; the envelope spectrum should recover 76 Hz.
"""

import numpy as np

from paderborn.spectra import envelope_spectrum, fft_spectrum, spectrogram_db, welch_psd

FS = 64_000
FAULT_HZ = 76.0
RESONANCE_HZ = 3200.0


def bearing_like_signal(seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    impacts = np.zeros_like(t)
    impacts[(np.arange(0, seconds, 1 / FAULT_HZ) * FS).astype(int)] = 1.0
    ringing = np.exp(-t[:400] * 2000) * np.sin(2 * np.pi * RESONANCE_HZ * t[:400])
    x = np.convolve(impacts, ringing)[: len(t)]
    return x + 0.05 * rng.standard_normal(len(t))


def peak_freq(f, a, fmin, fmax):
    mask = (f > fmin) & (f < fmax)
    return f[mask][np.argmax(a[mask])]


def test_fft_peak_near_resonance():
    f, a = fft_spectrum(bearing_like_signal(), FS)
    assert abs(peak_freq(f, a, 1000, 10_000) - RESONANCE_HZ) < 200


def test_envelope_recovers_fault_frequency():
    f, a = envelope_spectrum(bearing_like_signal(), FS, band=(2500, 4000))
    assert abs(peak_freq(f, a, 20, 200) - FAULT_HZ) < 2


def test_window_controls_resolution():
    x = bearing_like_signal(seconds=1.0)
    f_full, _ = fft_spectrum(x, FS)
    f_short, _ = fft_spectrum(x, FS, window_s=0.25)
    assert np.isclose(f_full[1], 1.0, atol=0.01)
    assert np.isclose(f_short[1], 4.0, atol=0.01)


def test_welch_and_spectrogram_shapes():
    x = bearing_like_signal()
    f, p = welch_psd(x, FS, nperseg=4096)
    assert f.shape == p.shape and np.all(np.isfinite(p))
    f, t, s = spectrogram_db(x, FS, nperseg=1024)
    assert s.shape == (len(f), len(t))


def test_bad_band_rejected():
    import pytest

    with pytest.raises(ValueError):
        envelope_spectrum(bearing_like_signal(), FS, band=(4000, 2500))
