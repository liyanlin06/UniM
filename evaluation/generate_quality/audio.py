#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purely statistical audio quality scoring v4.7 (no model, no training).
Targeted updates:
A) Music-friendly SNR: add HPSS-assisted snr_db_hpss and use it only for music.
B) Low-contrast noisy-texture penalty: strongly penalize low-SNR samples with very low contrast.
C) Pure-music compensation bonus: add a mild contrast/chroma bonus when SNR is low.

Target ordering:
aud1 > aud2 > aud1_with_noise > aud2_with_noise > aud1-cut
"""
import argparse, json
import numpy as np
import librosa as li
import pyloudnorm as pyln
from scipy.signal import stft, get_window

# ---------------- Basic utilities (same as v4.6) ----------------
def load_and_preprocess(path, target_sr=48000):
    x, sr = li.load(path, sr=None, mono=False)
    if x.ndim == 2: x = np.mean(x, axis=0)
    if sr != target_sr:
        x = li.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    x = x - np.mean(x)
    idx = np.flatnonzero(np.abs(x) > 1e-4)
    if idx.size > 0:
        pad = int(0.2 * sr)
        x = x[max(0, idx[0] - pad): min(len(x), idx[-1] + pad)]
    return x.astype(np.float32), sr

def stft_mag(x, sr, n_fft=2048, hop=512, win='hann'):
    w = get_window(win, n_fft, fftbins=True)
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                   window=w, padded=True, boundary="zeros")
    return f, t, np.abs(Z)

def spectral_flatness_frames(S, eps=1e-12):
    gmean = np.exp(np.mean(np.log(S + eps), axis=0))
    amean = np.mean(S + eps, axis=0)
    sf_frames = gmean / amean
    sf_global = float(np.median(sf_frames))
    return sf_global, sf_frames

def percentile_bandwidth(freqs, power, p=0.95):
    psd = power / (np.sum(power) + 1e-12)
    cdf = np.cumsum(psd)
    idx = np.searchsorted(cdf, p)
    return float(freqs[min(idx, len(freqs)-1)])

def dynamic_range_db(x, sr, frame=0.05):
    hop = int(frame*sr/2); win = int(frame*sr)
    if win < 32: win = 32
    if hop < 16: hop = 16
    rms_db = []
    for i in range(0, len(x)-win, hop):
        seg = x[i:i+win]
        rms = np.sqrt(np.mean(seg**2))+1e-12
        rms_db.append(20*np.log10(rms))
    if not rms_db: return 0.0
    lo, hi = np.percentile(rms_db, [5, 95])
    return float(hi - lo)

def silence_ratio(x, sr, thr_db=-60):
    frame = int(0.02*sr); hop = int(0.01*sr)
    if len(x) < frame: return 0.0
    cnt = 0; tot = 0
    for i in range(0, len(x)-frame, hop):
        seg = x[i:i+frame]
        db = 20*np.log10(np.sqrt(np.mean(seg**2))+1e-12)
        cnt += (db < thr_db); tot += 1
    return float(cnt/max(1,tot))

def hard_clip_ratio(x, thr=0.995, run=4):
    hits = np.abs(x) >= thr
    if not np.any(hits): return 0.0
    count = 0; consec = 0
    for h in hits:
        consec = consec + 1 if h else 0
        if consec == run:
            count += 1; consec = 0
    return count / (len(x)/max(1,run))

# ---------- Mid-segment gap / long-silence penalty (same v4.6 baseline) ----------
def mid_gap_penalty_adaptive(x, sr,
                             frame_s=0.02, hop_s=0.01,
                             head_tail_s=0.2,
                             min_gap_s=0.10,
                             max_penalty_drop=0.98,
                             zero_thr=1e-4, zero_prop_thr=0.97):
    win = int(frame_s*sr); hop = int(hop_s*sr)
    if len(x) < win: return 1.0, 0.0, 1.0, 0.0
    s = int(head_tail_s*sr); e = max(s+win, len(x)-s)
    mid = x[s:e]
    if len(mid) < win: return 1.0, 0.0, 1.0, 0.0

    idxs = range(0, len(mid)-win, hop)
    rms_db = []; zero_prop = []
    for i in idxs:
        seg = mid[i:i+win]
        rms = np.sqrt(np.mean(seg**2)) + 1e-12
        rms_db.append(20*np.log10(rms))
        zero_prop.append(float(np.mean(np.abs(seg) < zero_thr)))
    rms_db = np.array(rms_db); zero_prop = np.array(zero_prop)
    if rms_db.size == 0: return 1.0, 0.0, 1.0, 0.0

    k = max(1, int(0.3*len(rms_db)))
    ref = float(np.median(np.sort(rms_db)[-k:]))
    thr = np.clip(ref - 20.0, -55.0, -30.0)
    silent_energy = rms_db < thr
    silent_zero   = zero_prop > zero_prop_thr
    silent = np.logical_or(silent_energy, silent_zero)

    run = 0; max_run = 0
    for sflag in silent:
        run = run + 1 if sflag else 0
        if run > max_run: max_run = run
    max_s = (max_run * hop) / sr

    if max_s <= min_gap_s:
        gap_pen = 1.0
    else:
        T = (len(mid)/sr) * 0.8
        ratio = min(max_penalty_drop, max(0.0, (max_s - min_gap_s) / max(1e-9, T)))
        gap_pen = 1.0 - ratio

    active_ratio = float(np.mean(~silent))
    if active_ratio >= 0.85:
        cov_pen = 1.0
    else:
        cov_pen = max(0.1, 1.0 - 2.0*(0.85 - active_ratio))

    silent_prop = float(np.mean(silent))
    return float(np.clip(gap_pen * cov_pen, 0.0, 1.0)), float(max_s), float(cov_pen), float(silent_prop)

def band_score(val, lo, hi, soft=1.5):
    if val < lo:   s = np.exp(-(lo - val)/soft)
    elif val > hi: s = np.exp(-(val - hi)/soft)
    else:          s = 1.0
    return float(np.clip(s, 0, 1))

def logistic_score(x, x0=28.0, k=0.35):
    return float(1.0/(1.0+np.exp(-k*(x - x0))))

def weighted_geo_mean(scores, weights, eps=1e-9):
    s = np.clip(np.array(scores, dtype=float), eps, 1.0)
    w = np.array(weights, dtype=float)
    return float(np.exp(np.sum(w*np.log(s)) / np.sum(w)))

# ---------- High-frequency hiss slope (4–12 kHz) ----------
def hiss_penalty(f, mean_psd, f_lo=4000, f_hi=12000):
    mask = (f >= f_lo) & (f <= f_hi)
    if np.sum(mask) < 10:
        return 1.0, 0.0
    x = np.log10(f[mask] + 1e-9)
    y = np.log10(mean_psd[mask] + 1e-12)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    if slope <= -0.5:
        return 1.0, float(slope)
    t = (slope + 0.5) / (0.15 + 0.5)
    pen = 1.0 / (1.0 + 6.0 * t)
    return float(np.clip(pen, 0.25, 1.0)), float(slope)

def band_energy_ratio(f, psd, f_lo, f_hi):
    mask = (f >= f_lo) & (f <= f_hi)
    num = np.sum(psd[mask])
    den = np.sum(psd) + 1e-12
    return float(num / den)

def periodicity_score(x, sr, fmin=50, fmax=400, frame_ms=30, hop_ms=10):
    win = int(sr * frame_ms / 1000); hop = int(sr * hop_ms / 1000)
    if win < 128: win = 128
    if hop < 64: hop = 64
    lag_min = int(sr / fmax); lag_max = int(sr / fmin)
    peaks = []
    for i in range(0, len(x) - win, hop):
        seg = x[i:i+win].astype(np.float64)
        seg = seg - np.mean(seg)
        en = np.sum(seg**2) + 1e-12
        ac = np.correlate(seg, seg, mode='full')[win-1 : win-1+lag_max+1]
        ac = ac / en
        if lag_max <= lag_min or len(ac) <= lag_min+1: continue
        peaks.append(np.max(ac[lag_min:lag_max+1]))
    if not peaks: return 0.0, 0.0
    med = float(np.median(peaks))
    s = (med - 0.25) / (0.8 - 0.25)
    return float(np.clip(s, 0.0, 1.0)), med

def dr_score_mono(dr_db):
    return float(1.0 / (1.0 + np.exp(-(dr_db - 8.0) / 2.0)))

def spectral_contrast_score(S, sr):
    try:
        contrast = li.feature.spectral_contrast(S=S, sr=sr)
        val = float(np.median(np.median(contrast, axis=1)))
        sc = (val - 15.0) / (40.0 - 15.0)
        return float(np.clip(sc, 0.0, 1.0)), val
    except Exception:
        return 0.0, 0.0

def chroma_structure_score(x, sr):
    try:
        C = li.feature.chroma_cqt(y=x, sr=sr)
        var_time = np.var(C, axis=1)
        val = float(np.median(var_time))
        sc = (val - 0.005) / (0.05 - 0.005)
        return float(np.clip(sc, 0.0, 1.0)), val
    except Exception:
        return 0.0, 0.0

# ---- Spectral-flatness frame SNR + soft noisy-frame ratio (same as v4.6) ----
def music_friendly_snr(rms_vals, rms_db, sf_frames):
    valid = rms_db > -50
    if np.sum(valid) < 20:
        return None, 0.0
    sf = sf_frames[:len(valid)][valid] if len(sf_frames) >= len(valid) else sf_frames
    rv = rms_vals[valid]
    s_mask = sf < 0.45
    n_mask = sf > 0.65
    min_frames = max(10, int(0.05 * len(sf)))
    snr_db_sf = None
    if np.sum(s_mask) >= min_frames and np.sum(n_mask) >= min_frames:
        sig = np.median(rv[s_mask])
        noi = np.median(rv[n_mask])
        snr_db_sf = 20*np.log10((sig + 1e-12)/(noi + 1e-12))
    noisy_prop_soft = float(np.mean(sf > 0.55))
    return snr_db_sf, noisy_prop_soft

# === v4.7 addition: HPSS-assisted SNR (music only) ===
def hpss_snr_db(S):
    try:
        H, P = li.decompose.hpss(S)  # Input magnitude spectrogram
        HP = H + P
        R = np.maximum(S - HP, 0.0)
        num = np.sum(HP**2)
        den = np.sum(R**2) + 1e-12
        return float(10.0 * np.log10(num/den))
    except Exception:
        return None

# ---------------- Main pipeline ----------------
def score_audio(path):
    x, sr = load_and_preprocess(path, target_sr=48000)

    f, t, S = stft_mag(x, sr, n_fft=2048, hop=512)
    P = S**2
    mean_psd = np.mean(P, axis=1)

    sf_global, sf_frames = spectral_flatness_frames(S)
    structure = np.clip(1.0 - sf_global, 0, 1)

    f95 = percentile_bandwidth(f, mean_psd, p=0.95)
    bw_raw = np.clip(f95 / (0.9*(sr/2)), 0, 1)

    # Frame-level energy
    frame = int(0.02*sr); hop = int(0.01*sr)
    rms_vals, rms_db = [], []
    for i in range(0, len(x)-frame, hop):
        seg = x[i:i+frame]
        rms = np.sqrt(np.mean(seg**2))+1e-12
        rms_vals.append(rms)
        rms_db.append(20*np.log10(rms))
    if not rms_vals: rms_vals=[1e-12]; rms_db=[-120.0]
    rms_vals = np.array(rms_vals); rms_db = np.array(rms_db)

    # Original SNR estimate
    min_len = min(len(rms_vals), len(sf_frames))
    rms_vals = rms_vals[:min_len]; rms_db = rms_db[:min_len]; sf_frames = sf_frames[:min_len]
    mask_noise = (rms_db < -45) & (sf_frames > 0.6)
    if np.sum(mask_noise) >= 10:
        noise = np.percentile(rms_vals[mask_noise], 20)
    else:
        noise = np.percentile(rms_vals, 10)
    snr_db_orig = 20*np.log10((np.mean(rms_vals)+1e-12)/(noise+1e-12))
    snr_db_orig = float(np.clip(snr_db_orig, -10, 50))

    # Spectral-flatness frame SNR + soft noisy-frame ratio
    snr_db_sf, noisy_prop_soft = music_friendly_snr(rms_vals, rms_db, sf_frames)

    # Spectral features
    contrast_score, contrast_db = spectral_contrast_score(S, sr)
    chroma_score, chroma_var = chroma_structure_score(x, sr)
    periodic_score, periodic_peak = periodicity_score(x, sr)

    # Loose music detection
    is_music = (contrast_score >= 0.22) or (chroma_score >= 0.6)

    # === v4.7: HPSS-assisted SNR (music only) ===
    snr_db_hpss = hpss_snr_db(S) if is_music else None

    # Use the most favorable SNR among the original, SF-based, and HPSS-based estimates
    cand = [snr_db_orig]
    if snr_db_sf is not None: cand.append(snr_db_sf)
    if snr_db_hpss is not None: cand.append(snr_db_hpss)
    snr_db_eff = float(np.clip(max(cand), -10, 50))

    # Music-friendly mapping threshold
    snr_x0 = 22.0 if is_music else 28.0
    snr_score = logistic_score(snr_db_eff, x0=snr_x0, k=0.33)

    # DR & LUFS
    dr_db = dynamic_range_db(x, sr)
    dr_score = dr_score_mono(dr_db)
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(x.astype(np.float64))
    lufs_score = band_score(lufs, lo=-22, hi=-14, soft=2.0)

    # Level, transient, and crest metrics
    clip_r = hard_clip_ratio(x, thr=0.995, run=4)
    dc = float(np.abs(np.mean(x)))
    sil_r = silence_ratio(x, sr, thr_db=-60)
    clip_sc = np.clip(1.0 - (clip_r / 0.02), 0, 1)
    dc_sc   = np.clip(1.0 - (dc / 1e-3), 0, 1)
    sil_sc  = np.clip(1.0 - (sil_r / 0.4), 0, 1)
    level_score = float(np.mean([clip_sc, dc_sc, sil_sc]))

    w = max(8, int(sr * 10 / 1000))
    env = np.convolve(np.abs(x), np.ones(w)/w, mode='same')
    d_env = np.abs(np.diff(env))
    thr95 = np.percentile(d_env, 95)
    tr = float(np.mean(d_env > thr95))
    transient_score = np.clip(1.0 - (tr / 0.2), 0, 1)

    rms_all = np.sqrt(np.mean(x**2))+1e-12
    crest_db = 20*np.log10((np.max(np.abs(x))+1e-12)/rms_all)
    crest_score = band_score(crest_db, lo=10, hi=18, soft=1.0)

    enable_periodic_bonus = (structure >= 0.5) and (snr_db_eff >= 15.0)

    # Noise / hiss penalties
    penalty_noise = 1.0 / (1.0 + 10.0 * float(noisy_prop_soft))
    hf_slope, penalty_hiss = None, None
    penalty_hiss, hf_slope = hiss_penalty(f, mean_psd, f_lo=4000, f_hi=12000)
    hf_ratio = band_energy_ratio(f, mean_psd, 7000, 16000)
    if structure < 0.4:
        excess = max(0.0, hf_ratio - 0.22)
        penalty_hf_ratio = 1.0 / (1.0 + 8.0 * excess)
    else:
        penalty_hf_ratio = 1.0

    # Mid-segment gap / blank-region penalty (same as v4.6), plus extra silence suppression
    gap_cov_pen, max_gap_s, cov_pen, silent_prop = mid_gap_penalty_adaptive(
        x, sr, frame_s=0.02, hop_s=0.01,
        head_tail_s=0.2, min_gap_s=0.10, max_penalty_drop=0.98,
        zero_thr=1e-4, zero_prop_thr=0.97
    )
    if silent_prop > 0.35:
        gap_cov_pen *= 1.0 / (1.0 + 4.0 * (silent_prop - 0.35))
    gap_cov_pen = float(np.clip(gap_cov_pen, 0.0, 1.0))

    # === v4.7 addition: low-contrast noisy-texture penalty ===
    penalty_low_contrast = 1.0
    if snr_db_eff < 18.0:
        deficit = max(0.0, 0.12 - float(contrast_score))
        if deficit > 0.0:
            penalty_low_contrast = 1.0 / (1.0 + 3.2 * deficit)

    # === v4.7 addition: music compensation bonus (only when SNR is low) ===
    bonus_music = 1.0
    if is_music and snr_db_eff < 18.0:
        bonus_music = 1.0 + 0.12*float(contrast_score) + 0.08*float(chroma_score)

    # Aggregate scores
    scores = [
        snr_score, structure, dr_score, lufs_score,
        level_score, transient_score, crest_score,
        max(0.01, 0.8 + 0.2*bw_raw),
        max(0.01, contrast_score),
        max(0.01, chroma_score)
    ]
    weights = [3.5, 2.5, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 2.4, 1.2]
    base = weighted_geo_mean(scores, weights)

    penalty_clip = 1.0 / (1.0 + 400.0 * clip_r)
    bonus_periodic = (1.0 + 0.15 * periodic_score) if enable_periodic_bonus else 1.0

    q01 = base * penalty_clip * gap_cov_pen * penalty_noise * penalty_hiss * penalty_hf_ratio
    q01 *= penalty_low_contrast       # v4.7 addition: low-contrast penalty
    q01 *= bonus_music                # v4.7 addition: music compensation
    q01 = float(np.clip(q01, 0.0, 1.0))
    Q = float(100.0 * (q01 ** 0.7))

    return {
        "quality_0_100": round(Q, 2),
        "q01": round(q01, 4),
        # SNR-related statistics
        "snr_db_eff": round(snr_db_eff, 2),
        "snr_db_orig": round(snr_db_orig, 2),
        "snr_db_sf": None if snr_db_sf is None else round(snr_db_sf, 2),
        "snr_db_hpss": None if snr_db_hpss is None else round(snr_db_hpss, 2),  # v4.7
        "snr_score": round(snr_score, 3),
        "is_music": bool(is_music),
        # Structure / dynamics / level
        "structure(1-SF)": round(structure, 3),
        "dr_db": round(dr_db, 2), "dr_score": round(dr_score, 3),
        "lufs": round(lufs, 2), "lufs_score": round(lufs_score, 3),
        "level_score": round(level_score, 3),
        "transient_score": round(transient_score, 3),
        "crest_db": round(crest_db, 2), "crest_score": round(crest_score, 3),
        "bandwidth_raw": round(bw_raw, 3),
        "periodic_score": round(periodic_score, 3),
        "periodic_peak": round(periodic_peak, 3),
        # Music structure
        "contrast_score": round(contrast_score, 3),
        "contrast_db": round(contrast_db, 2),
        "chroma_score": round(chroma_score, 3),
        "chroma_var": float(f"{chroma_var:.6f}"),
        # Noise / hiss
        "noisy_prop_soft": round(float(noisy_prop_soft), 3),
        "hf_slope_logP": round(hf_slope, 3),
        "hf_ratio_7_16k": round(hf_ratio, 3),
        # Penalties / bonuses
        "penalty_clip": round(penalty_clip, 3),
        "gap_cov_penalty": round(gap_cov_pen, 3),
        "penalty_noise": round(penalty_noise, 3),
        "penalty_hiss": round(penalty_hiss, 3),
        "penalty_hf_ratio": round(penalty_hf_ratio, 3),
        "penalty_low_contrast": round(penalty_low_contrast, 3),   # v4.7
        "bonus_music": round(bonus_music, 3),                     # v4.7
        "bonus_periodic": round(bonus_periodic, 3),
        # Other diagnostics
        "max_mid_gap_seconds": round(max_gap_s, 3),
        "silent_prop_mid": round(silent_prop, 3),
        "clip_ratio_hard": round(hard_clip_ratio(x), 6),
        "silence_ratio_global": round(silence_ratio(x, sr), 3),
        "dc_offset": float(f"{np.abs(np.mean(x)):.6e}"),
    }

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Purely statistical audio quality scoring v4.7 (no model, no training)")
    ap.add_argument("input", help="Input audio file (mp3/wav/m4a/flac/...)")
    ap.add_argument("--json", action="store_true", help="Output JSON with component metrics")
    args = ap.parse_args()
    out = score_audio(args.input)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out["quality_0_100"])

if __name__ == "__main__":
    main()
