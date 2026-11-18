#!/usr/bin/env python3
"""
generate_ltass_noise_wav.py

Synthesizes a calibrated, seamlessly loopable/circular 8.0-second (352,800 samples
at 44.1 kHz, 24-bit mono PCM) speech-shaped noise file ``Stimuli/ltass_noise.wav``
matched in exact LTASS spectrum and 0-dB-SNR RMS to ``Stimuli/test/*.wav``.

Calibration reference:
  - In ``Stimuli/practice/*_60SNR.wav`` (x) and ``Stimuli/test/*_<SNR>SNR.wav`` (y),
    y = s + 10^(-SNR/20) * n_0 and x = s + 10^(-3) * n_0, where 0.01 RMS == 60 dB SPL.
  - Subtracting x from y across the 0-dB-SNR stimuli isolates the exact empirical
    LTASS noise reference n_0 (sr = 44100 Hz, 0-dB-SNR RMS = 0.010178).
  - Synthesizing on the exact N = 352,800 rFFT grid makes ``ltass_noise.wav``
    strictly C^infinity-periodic across the buffer boundary, so circular slicing
    ``(offset + k) % N`` for any random trial offset has zero wrap discontinuity.
"""

from __future__ import annotations

from pathlib import Path
import wave
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

AUD_FREQS_HZ = np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000], dtype=float)
ANSI_RETSPL_DB = np.array([18.0, 11.0, 8.5, 7.5, 7.5, 9.0, 10.0, 9.5, 15.5, 13.0], dtype=float)


def erb_band_hl(sig: np.ndarray, sr: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(len(sig), 1.0 / float(sr))
    psd = (np.abs(np.fft.rfft(sig)) ** 2) / (len(sig) ** 2) * 2.0
    out = np.zeros(len(AUD_FREQS_HZ), dtype=float)
    for i, f0 in enumerate(AUD_FREQS_HZ):
        erb = 24.7 * (4.37 * (f0 / 1000.0) + 1.0)
        m = (freqs >= f0 - 0.5 * erb) & (freqs <= f0 + 0.5 * erb)
        pwr = max(float(np.sum(psd[m])), 1e-20)
        out[i] = 60.0 + 10.0 * np.log10(pwr / (0.01 ** 2)) - ANSI_RETSPL_DB[i]
    return out


def write_pcm24_mono_wav(path: Path, sr: int, samples_f64: np.ndarray) -> None:
    clipped = np.clip(samples_f64, -0.999999, 0.999999)
    ints = np.round(clipped * 8388608.0).astype(np.int32)
    # Pack signed 24-bit little-endian PCM (3 bytes/sample):
    raw_u32 = (ints & 0xFFFFFF).astype("<u4")
    b_view = raw_u32.view(np.uint8).reshape(-1, 4)[:, :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(b_view.tobytes())


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    stim_dir = repo_root / "Stimuli"

    # 1. Extract exact 0-dB-SNR noise reference & mean token RMS across all 20 pairs:
    sr_ref = 44100
    rms_list: list[float] = []
    longest_n0: np.ndarray | None = None

    for p_path in sorted((stim_dir / "practice").glob("VCV_*_60SNR.wav")):
        stem = p_path.name.replace("_60SNR.wav", "")
        matches = list((stim_dir / "test").glob(f"{stem}_*SNR.wav"))
        if not matches:
            continue
        t_path = matches[0]
        snr_nom = float(t_path.name.split("_")[-1].replace("SNR.wav", ""))
        sr, x_i = wavfile.read(str(p_path))
        _, y_i = wavfile.read(str(t_path))
        sr_ref = int(sr)
        scale = float(np.iinfo(x_i.dtype).max) + 1.0
        x = x_i.astype(np.float64) / scale
        y = y_i.astype(np.float64) / scale
        g = 10.0 ** (-snr_nom / 20.0)
        n_tok = (y - x) / (g - 0.001)
        rms_list.append(float(np.sqrt(np.mean(n_tok ** 2))))
        if longest_n0 is None or len(n_tok) > len(longest_n0):
            longest_n0 = n_tok

    assert longest_n0 is not None
    target_rms = float(np.mean(rms_list))

    # 2. Smooth Welch PSD of empirical LTASS noise (nperseg=2048 -> ~21.5 Hz resolution):
    f_welch, pxx = welch(
        longest_n0,
        fs=sr_ref,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        detrend="constant",
        scaling="density",
    )

    # 3. Synthesize 8.0-second circularly periodic stationary LTASS noise:
    n_samples = 8 * sr_ref  # 352,800 samples = 8.000 s
    f_grid = np.fft.rfftfreq(n_samples, 1.0 / float(sr_ref))
    psd_interp = np.interp(f_grid, f_welch, np.maximum(pxx, 1e-24))
    psd_interp[0] = 0.0  # Zero DC offset
    # Gentle sub-35 Hz highpass guard to prevent sub-audible transducer excursion:
    hp_guard = np.clip((f_grid - 20.0) / 25.0, 0.0, 1.0) ** 2
    psd_interp *= hp_guard

    rng = np.random.default_rng(20261118)
    re = rng.standard_normal(len(f_grid))
    im = rng.standard_normal(len(f_grid))
    im[0] = 0.0
    if n_samples % 2 == 0:
        im[-1] = 0.0
    spec = (re + 1j * im) * np.sqrt(psd_interp)
    noise = np.fft.irfft(spec, n=n_samples)
    noise -= np.mean(noise)
    noise *= target_rms / float(np.sqrt(np.mean(noise ** 2)))

    out_path = stim_dir / "ltass_noise.wav"
    write_pcm24_mono_wav(out_path, sr_ref, noise)

    # 4. Verify read-back & ERB band parity against empirical 0-dB-SNR reference:
    sr_chk, n_chk_i = wavfile.read(str(out_path))
    n_chk = n_chk_i.astype(np.float64) / (float(np.iinfo(n_chk_i.dtype).max) + 1.0)
    erb_ref = erb_band_hl(longest_n0, sr_ref)
    erb_syn = erb_band_hl(n_chk, int(sr_chk))
    max_erb_diff = float(np.max(np.abs(erb_syn - erb_ref)))

    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Sample rate : {sr_chk} Hz, samples: {len(n_chk)} ({len(n_chk)/sr_chk:.2f} s), 24-bit mono PCM")
    print(f"  Target RMS  : {target_rms:.7f} | Readback RMS: {np.sqrt(np.mean(n_chk**2)):.7f} | Peak: {np.max(np.abs(n_chk)):.5f}")
    print(f"  Peak @ -18 dB SNR: {np.max(np.abs(n_chk)) * (10**(18/20)):.4f} (<= 1.0 full scale)")
    print(f"  Max |ERB band HL (synth - empirical)| across 250..8000 Hz: {max_erb_diff:.3f} dB")
    for f0, r_v, s_v in zip(AUD_FREQS_HZ, erb_ref, erb_syn):
        print(f"    {int(f0):5d} Hz: ref={r_v:6.2f} dB HL | synth={s_v:6.2f} dB HL (diff={s_v - r_v:+5.2f} dB)")


if __name__ == "__main__":
    main()
