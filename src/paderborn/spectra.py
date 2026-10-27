"""Spectral analysis tools for bearing vibration and current signals.

All functions take a 1D signal ``x`` sampled at ``fs`` Hz and return arrays
ready to plot. They are kept free of any GUI code so they can be tested and
reused in notebooks and in the ML feature pipeline.

    fft_spectrum        amplitude spectrum of one Hann-windowed segment
    welch_psd           averaged power spectral density in dB (less noisy)
    envelope_spectrum   band-pass -> Hilbert envelope -> spectrum
                        (the standard bearing-diagnostics tool: reveals the
                        fault repetition rate hidden in a resonance band)
    spectrogram_db      short-time Fourier transform, frequency over time
"""

import numpy as np
from scipy import signal


def _segment(x, fs, window_s):
    n = len(x) if window_s is None else min(len(x), int(window_s * fs))
    return np.asarray(x[:n], dtype=np.float64)


def fft_spectrum(x, fs, window_s=None):
    """Single-sided amplitude spectrum of the first ``window_s`` seconds."""
    seg = _segment(x, fs, window_s)
    seg = seg - seg.mean()
    w = np.hanning(len(seg))
    amp = 2 * np.abs(np.fft.rfft(seg * w)) / w.sum()
    return np.fft.rfftfreq(len(seg), 1 / fs), amp


def welch_psd(x, fs, nperseg=8192):
    """Welch-averaged power spectral density in dB (re 1 unit^2/Hz)."""
    f, pxx = signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)), window="hann")
    return f, 10 * np.log10(pxx + 1e-20)


def envelope_spectrum(x, fs, band, window_s=None, order=4):
    """Envelope spectrum: band-pass around a resonance, take |Hilbert|, then FFT.

    Bearing faults produce short impacts that ring a structural resonance.
    The fault frequency appears as the repetition rate of those impacts,
    i.e. as an amplitude modulation of the resonance band. Demodulating
    the band exposes it as clean peaks at BPFO / BPFI and their harmonics.
    """
    low, high = band
    nyq = fs / 2
    if not 0 < low < high < nyq:
        raise ValueError(f"band {band} must satisfy 0 < low < high < {nyq:.0f} Hz")
    seg = _segment(x, fs, window_s)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, seg)
    envelope = np.abs(signal.hilbert(filtered))
    return fft_spectrum(envelope, fs)


def spectrogram_db(x, fs, nperseg=4096):
    """STFT power in dB. Returns (freqs, times, power[freq, time])."""
    f, t, sxx = signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann"
    )
    return f, t, 10 * np.log10(sxx + 1e-20)
