"""
auto_scan.py -- the full automated grating scan (attenuation-outer / angle-inner).

Implements the procedure in `scan_flowchart_v2`:

  * the mirror order-sorter sweeps attenuation MONOTONICALLY, deepest -> open, and
    NEVER returns to a previous setting -- the only motion the non-repeatable,
    un-encoded PIM05 can be trusted to make;
  * at each attenuation level the mirror is held still while the sample rotates
    through every grazing angle (the cheap, repeatable motion);
  * a grating-OUT I0 (direct-beam) shot brackets each level for normalization/drift;
  * per angle we STITCH the levels where that angle is on-scale (ratios fit from
    overlaps -- no mirror position needed) and I0-normalize to grating efficiency.

Two error sources, two knobs:
  * pileup error  -> mirror attenuation, holding the brightest on-scale feature
                     under `cap_pct` pileup (the deadtime-correctable regime);
  * S/N error     -> exposure time sized to a per-strip `floor_pct` statistical
                     floor, plus the stitch that equalizes S/N across the pattern.
  * `bracket_b`   -> attenuation ratio between levels (~e..10; see bracket-ratio note).

This is an offline de-risking model, not beamline control code.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics
import sample
from detector import MythenDetector

COATING = "Pt"                 # BL 5.3.1 grating; brightness follows Pt reflectivity vs angle
SAT_PILEUP = 0.15              # a strip above this pileup is "saturated" (correction unreliable)


def _levels(A_deep, bracket_b):
    """Attenuation ladder from the deepest setting (direct beam at cap) opening in
    x`bracket_b` steps up to full transmission.  Never returns; monotonic."""
    K = int(np.ceil(np.log(1.0 / A_deep) / np.log(bracket_b))) + 1
    A = [min(A_deep * bracket_b ** k, 1.0) for k in range(K)]
    out = []
    for a in A:
        out.append(a)
        if a >= 1.0:
            break
    # merge near-duplicate levels (e.g. 0.99 and 1.0) so we don't waste a sweep
    ded = [out[0]]
    for a in out[1:]:
        if a / ded[-1] >= 1.5:
            ded.append(a)
        else:
            ded[-1] = a
    return ded


def run_auto_scan(E_keV=8.0, angles=None, cap_pct=10.0, floor_pct=0.5, bracket_b=10.0,
                  slit_um=100.0, L_m=0.698, coating=COATING, satellite_frac=0.0016,
                  step_jitter=0.15, seed=1):
    """Run the level-outer automated scan.  Returns a results dict.

    cap_pct   : pileup cap the brightest on-scale feature is held under.
    floor_pct : target per-strip statistical floor (S/N = 100/floor_pct) on the
                orders -- sets the per-level exposure.
    bracket_b : attenuation ratio between adjacent levels.
    step_jitter: fractional non-repeatability of each open-loop mirror step (the
                 ladder ratios come out imperfect -- the stitch fits them blind).
    """
    if angles is None:
        angles = np.round(np.arange(0.20, 0.72, 0.05), 3)     # shallow -> steep
    rng = np.random.default_rng(seed)
    sigma = physics.beam_sigma_strips(slit_um, E_keV, L_m)
    det = MythenDetector(energy_keV=E_keV, seed=seed, beam_sigma_ch=sigma)
    tau = det.tau
    R_cap = physics.true_rate_for_pileup(cap_pct / 100.0, E_keV)     # true-rate cap, cps/strip
    R_rel = physics.true_rate_for_pileup(SAT_PILEUP, E_keV)          # reliability cut
    F0 = physics.incident_flux(E_keV)                               # peak-strip direct rate at A=1
    N_inc_full = F0 / det.peak_frac                                 # total beam at A=1

    # exposure per level: reach the floor S/N on a feature sitting at the cap
    s_min = 100.0 / floor_pct
    t_level = max(s_min ** 2 / R_cap, 0.05)                          # seconds
    frame_s = 0.1
    n_frames = int(np.ceil(t_level / frame_s))

    # attenuation ladder: deepest = direct beam at the cap
    A_deep = R_cap / F0
    A_target = _levels(A_deep, bracket_b)
    # realize with the NON-REPEATABLE mirror: each open step lands imperfectly
    A_real, a = [], A_deep
    for k, At in enumerate(A_target):
        if k == 0:
            a = At                                                  # level 0 feedback-set (exact)
        else:
            ratio = (At / A_target[k - 1]) * (1.0 + rng.uniform(-step_jitter, step_jitter))
            a = min(a * ratio, 1.0)
        A_real.append(a)
    A_real = np.array(A_real)

    nA = len(A_real)
    na = len(angles)
    nch = det.nch

    # storage: per (level, angle) corrected rate, counts, saturation mask
    rate = np.zeros((nA, na, nch)); counts = np.zeros((nA, na, nch)); sat = np.zeros((nA, na, nch), bool)
    I0 = np.zeros(nA); I0_saturated = np.zeros(nA, bool)

    for k, A in enumerate(A_real):
        # --- grating OUT: I0 (direct beam total), clean only where on-scale ---
        direct_peak = A * F0
        I0[k] = A * N_inc_full                                       # true incident total at this level
        I0_saturated[k] = direct_peak > R_rel                        # direct beam saturates when too open
        # --- grating IN: rotate through every angle at this held mirror level ---
        for j, alpha in enumerate(angles):
            true_strip = sample.diffraction_pattern(
                E_keV, alpha, A * N_inc_full, det, peak_sigma_strip=sigma,
                material=coating, satellite_frac=satellite_frac)
            det.acquire_pattern(true_strip, n_frames=n_frames, frame_s=frame_s)
            summed = det.frames.sum(axis=0).astype(float)
            M = summed / (n_frames * frame_s)
            Ncorr = det.correct_rate(M)
            rate[k, j] = Ncorr
            counts[k, j] = summed
            sat[k, j] = Ncorr >= R_rel                               # saturated / unreliable

    # ---- assemble per angle: stitch its valid levels, I0-normalize ----
    eff = np.full((na, nch), np.nan)          # grating efficiency per strip
    best_counts = np.zeros((na, nch))
    single_counts = np.zeros((na, nch))       # a single deepest-level exposure of equal total time
    scales_all = []
    for j in range(na):
        # monotonic-saturation guard: once a strip saturates going deep->open it stays out
        cum_sat = np.zeros(nch, bool)
        valid = np.zeros((nA, nch), bool)
        for k in range(nA):
            cum_sat |= sat[k, j]
            valid[k] = (~cum_sat) & (counts[k, j] > 0)
        # stitch: reference = deepest level (k=0), fold brighter levels via overlap ratio
        ref = rate[0, j].copy(); scales = [1.0]
        for k in range(1, nA):
            ov = valid[k] & valid[k - 1] & (counts[k - 1, j] > 200)
            if ov.sum() >= 3:
                s = float(np.sum(ref[ov] * rate[k, j][ov]) / np.sum(rate[k, j][ov] ** 2))
            else:
                s = scales[-1]
            scales.append(s)
            ref = np.where(valid[k], s * rate[k, j], ref)
        scales_all.append(scales)
        # brightest valid level per strip -> stitched rate + its counts (best S/N)
        stit = np.full(nch, np.nan); bc = np.zeros(nch)
        for k in range(nA):
            use = valid[k] & (counts[k, j] > 0)
            stit[use] = scales[k] * rate[k, j][use]
            bc[use] = np.maximum(bc[use], counts[k, j][use])
        best_counts[j] = bc
        # I0-normalize -> efficiency (incident total cancels on the deepest reference)
        eff[j] = stit / I0[0]
        # single-exposure baseline: everything at the deepest level, same total time (nA*t_level)
        single_counts[j] = counts[0, j] * nA

    integ = np.array([np.nansum(np.where(eff[j] > 0, eff[j], 0)) for j in range(na)])  # rocking curve
    return dict(E=E_keV, angles=np.array(angles), cap_pct=cap_pct, floor_pct=floor_pct,
                bracket_b=bracket_b, A_target=np.array(A_target), A_real=A_real,
                I0=I0, I0_saturated=I0_saturated, n_levels=nA, t_level=t_level,
                rate=rate, counts=counts, sat=sat, eff=eff, best_counts=best_counts,
                single_counts=single_counts, integ=integ, scales=scales_all, det=det,
                coating=coating, satellite_frac=satellite_frac, sigma=sigma)


def _snr_range(counts_row, mask):
    sn = np.sqrt(np.maximum(counts_row[mask], 1e-9))
    return sn.max() / sn.min() if mask.sum() else np.nan


def make_figure(res, example_angle_idx=None, path="auto_scan.png"):
    """4-panel summary: ladder, one stitched angle, S/N uniformity, rocking curve."""
    ang = res["angles"]; nA = res["n_levels"]
    j = example_angle_idx if example_angle_idx is not None else int(np.argmin(np.abs(ang - 0.30)))
    ch = np.arange(res["det"].nch)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))

    # (a) attenuation ladder (target vs realized, non-repeatable)
    ax[0, 0].semilogy(range(nA), res["A_target"], "o-", label="target ladder")
    ax[0, 0].semilogy(range(nA), res["A_real"], "s--", label="realized (non-repeatable)")
    ax[0, 0].set_title(f"Attenuation ladder ({nA} levels, ×{res['bracket_b']:g}; mirror never returns)")
    ax[0, 0].set_xlabel("level (deepest → open)"); ax[0, 0].set_ylabel("mirror attenuation A")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, which="both", alpha=0.25)

    # (b) stitched efficiency vs true efficiency at the example angle (fidelity)
    effj = res["eff"][j]
    true_eff = sample.diffraction_pattern(res["E"], ang[j], 1.0, res["det"],
                                          peak_sigma_strip=res["sigma"], material=res["coating"],
                                          satellite_frac=res["satellite_frac"])
    ax[0, 1].semilogy(ch, np.clip(true_eff, 1e-7, None), "-", color="#333", lw=1.2, label="true efficiency")
    ax[0, 1].semilogy(ch, np.clip(effj, 1e-7, None), ".", ms=3, color="#c0392b", label="stitched (recovered)")
    ax[0, 1].set_title(f"Stitched vs true pattern at α = {ang[j]:.2f}°  (Pt-coated)")
    ax[0, 1].set_xlabel("detector strip"); ax[0, 1].set_ylabel("grating efficiency (per strip)")
    ax[0, 1].set_xlim(400, 640); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, which="both", alpha=0.25)

    # (c) S/N across the pattern: single deepest exposure vs stitched
    rep = effj > (np.nanmax(effj) * 1e-2)
    sn_single = np.sqrt(np.maximum(res["single_counts"][j], 1e-9))
    sn_stit = np.sqrt(np.maximum(res["best_counts"][j], 1e-9))
    srs = _snr_range(res["single_counts"][j], rep); srt = _snr_range(res["best_counts"][j], rep)
    ax[1, 0].semilogy(ch, np.clip(sn_single, 0.3, None), ".", ms=2, color="#e08a1e",
                      label=f"single exposure ({srs:.0f}× over orders)")
    ax[1, 0].semilogy(ch, np.clip(sn_stit, 0.3, None), ".", ms=2, color="#1f5fa8",
                      label=f"stitched levels ({srt:.0f}× over orders)")
    ax[1, 0].set_title(f"S/N across the pattern at α = {ang[j]:.2f}°")
    ax[1, 0].set_xlabel("detector strip"); ax[1, 0].set_ylabel("S/N per strip (√counts)")
    ax[1, 0].set_xlim(380, 640); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, which="both", alpha=0.25)

    # (d) rocking curve: integrated efficiency vs grazing angle
    ax[1, 1].semilogy(ang, np.clip(res["integ"], 1e-12, None), "o-", color="#2e8b57")
    ax[1, 1].set_title("Rocking curve: integrated efficiency vs grazing angle")
    ax[1, 1].set_xlabel("grazing angle (deg)"); ax[1, 1].set_ylabel("integrated efficiency (a.u.)")
    ax[1, 1].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Automated grating scan @ {res['E']:g} keV — cap {res['cap_pct']:g}%, "
                 f"floor {res['floor_pct']:g}%, ×{res['bracket_b']:g} brackets "
                 f"({nA} levels × {len(ang)} angles, {res['t_level']:.2f}s/level)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    res = run_auto_scan()
    print(f"levels={res['n_levels']}  angles={len(res['angles'])}  t/level={res['t_level']:.2f}s")
    print(f"realized ladder ratios (blind, non-repeatable): "
          f"{[round(a, 4) for a in res['A_real']]}")
    j = int(np.argmin(np.abs(res['angles'] - 0.30)))
    print(f"stitch scales at α=0.30°: {[round(s, 4) for s in res['scales'][j]]}")
    p = make_figure(res, path="auto_scan.png")
    print("wrote", p)
