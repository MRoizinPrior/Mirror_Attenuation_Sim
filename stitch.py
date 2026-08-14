"""
stitch.py -- multi-exposure "bracketing / stitching" of a diffraction pattern.

The problem (Howard Padmore): a diffraction pattern has a huge dynamic range.
In ONE exposure the peak might have 1e5 counts (signal-to-noise ~ sqrt(1e5) ~ 316)
while the tail has 1e1 counts (S/N ~ 3) -- a ~100x spread in data quality. Bad.

The fix: record the SAME pattern several times at intensities spaced ~10x apart
(by changing the mirror attenuation). Each exposure measures ~one decade well:
its bright region saturates the detector and is discarded, but its faint region
is boosted into good statistics. Stitching the valid decades back together gives
a pattern whose S/N varies only ~sqrt(bracket) instead of the full 100x.

Pileup here is SECONDARY -- it only sets the maximum usable rate (the saturation
cap) for each exposure. The headline question this module answers is:
    how faithfully can we stitch the exposures back into the true pattern,
    and how much does the S/N uniformity improve?

Convention:
  * The true pattern is a normalized shape S(x) with peak 1 (a Gaussian, or a
    sum of orders). "Reference scale" = rate when the peak sits exactly at the
    saturation cap R_max (this is exposure/level 0).
  * Level k is 10^k times brighter (10^k less attenuation). Its true rate is
    S(x) * R_max * bracket^k; wherever that exceeds R_max the detector saturates
    and those points are thrown away. So level k is valid where S(x) <= bracket^-k.
  * We do NOT assume we know the exact intensity ratios -- we FIT them from the
    overlap regions, exactly as on the beamline.

Deps: numpy, matplotlib, physics.py (for tau -> max rate).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics
from actuator import PIM05Actuator


# ==========================================================================
# 0) Set the bracketed intensities with a REAL mirror scan
# ==========================================================================
# Instead of assuming exact x10 intensity steps, pick the mirror angles that give
# ~x10 attenuation steps and drive the (non-repeatable, un-encoded) PIM05 to them.
# Because the attenuation A(theta) = R(theta)^2 is steep and the actuator's step
# size is uncalibrated, the delivered intensities come out only APPROXIMATELY x10
# -- exactly Howard's "we won't know the intensities". Level 0 (dimmest, peak at
# the cap) is taken as feedback-set and exact; the brighter levels are open-loop
# decade steps and land imperfectly. Returns the realized relative intensities.
def mirror_intensities(E=8.0, n_levels=4, bracket=10.0, pileup_pct=10.0,
                       step_miscal=1.08, actuator_seed=0):
    F0 = physics.incident_flux(E)
    R_max = physics.true_rate_for_pileup(pileup_pct / 100.0, E)
    A0 = R_max / F0                                    # attenuation putting level-0 peak at the cap

    def angle_for_A(A_target):                         # invert A(theta)=A_target (A falls with theta)
        lo, hi = 0.05, 1.2
        for _ in range(60):
            m = 0.5 * (lo + hi)
            # A(m) too high -> angle too small -> search larger theta (raise lo)
            lo, hi = (m, hi) if physics.attenuation(m, E) > A_target else (lo, m)
        return 0.5 * (lo + hi)

    # target angles for an ideal x10 ladder (brighter = larger A = smaller angle)
    thetas = [angle_for_A(min(A0 * bracket ** k, 0.98)) for k in range(n_levels)]

    # drive one non-repeatable actuator dim -> bright (angles decrease monotonically,
    # so one direction -> no backlash reversal). The controller assumes 0.5 urad/step
    # but the true step is miscalibrated, so the moves land off target.
    act = PIM05Actuator(theta0_deg=thetas[0], step_urad=0.5 * step_miscal, seed=actuator_seed)
    nominal_dps = np.degrees(0.5e-6)                   # deg per step the controller BELIEVES
    landed = [act.theta_deg]                           # level 0 = exact (feedback-set)
    for k in range(1, n_levels):
        act.step(int(round((thetas[k] - act.theta_deg) / nominal_dps)))
        landed.append(act.theta_deg)

    A_real = [physics.attenuation(th, E) for th in landed]
    intensity = [a / A_real[0] for a in A_real]        # relative to level 0
    return dict(target_theta=thetas, landed_theta=landed, intensity=intensity, R_max=R_max)


# ==========================================================================
# 1) The true (noise-free) pattern shape, peak normalized to 1
# ==========================================================================
def pattern_shape(x, x0=320.0, sigma_px=5.0, orders=1, order_spacing_px=40.0,
                  order_falloff=10.0):
    """Normalized pattern S(x) (peak 1). orders=1 -> single Gaussian; orders>1
    adds diffraction orders at +order_spacing, each order_falloff x weaker."""
    S = np.zeros_like(x, dtype=float)
    for m in range(orders):
        amp = order_falloff ** (-m)
        S += amp * np.exp(-0.5 * ((x - (x0 + m * order_spacing_px)) / sigma_px) ** 2)
    return S / S.max()


# ==========================================================================
# 2) Record one exposure: apply the saturation cap, then Poisson noise
# ==========================================================================
def record_exposure(S, R_max, intensity, t_s, rng):
    """Return (measured_rate, valid_mask) for one bracketed exposure.

    true rate = S * R_max * intensity ; valid where that <= R_max (unsaturated).
    measured counts = Poisson(true_rate * t) ; measured rate = counts / t."""
    true_rate = S * R_max * intensity
    valid = true_rate <= R_max                       # discard saturated points
    counts = rng.poisson(np.where(valid, true_rate, 0.0) * t_s)
    return counts / t_s, valid


# ==========================================================================
# 3) Stitch: fit the unknown scale factor between adjacent exposures
# ==========================================================================
def fit_scale(ref_rate, raw_rate, overlap):
    """Least-squares scale s that best maps raw onto the reference over the
    overlap region: minimize sum (ref - s*raw)^2  ->  s = <ref*raw>/<raw*raw>."""
    a, b = ref_rate[overlap], raw_rate[overlap]
    return float(np.sum(a * b) / np.sum(b * b))


def stitch(exposures, min_overlap_counts=100):
    """exposures: list of dicts with keys rate, valid, counts (level 0 first,
    brightest last). Returns (stitched_rate, cumulative_scales).

    At each x we use the BRIGHTEST exposure still valid there (most counts =
    best statistics), rescaled onto the level-0 reference by the fitted factors."""
    n = len(exposures)
    npix = len(exposures[0]["rate"])
    scales = [1.0]                                    # cumulative scale to reference, level 0 = 1
    ref = exposures[0]["rate"].copy()                 # running reference-scale estimate
    for k in range(1, n):
        # overlap: both this level and the previous reference are valid, and the
        # reference side still has enough counts to anchor the fit
        ov = (exposures[k]["valid"] & exposures[k - 1]["valid"]
              & (exposures[k - 1]["counts"] > min_overlap_counts))
        if ov.sum() < 3:
            scales.append(scales[-1]); continue
        s_rel = fit_scale(ref, exposures[k]["rate"], ov)   # ref ~ s_rel * raw_k
        scales.append(s_rel)
        # fold this level (rescaled) into the running reference where it is valid
        ref = np.where(exposures[k]["valid"], s_rel * exposures[k]["rate"], ref)

    # final assembly: brightest valid exposure at each pixel, on the reference scale
    stitched = np.full(npix, np.nan)
    for k in reversed(range(n)):                       # brightest level first wins
        use = exposures[k]["valid"] & (exposures[k]["counts"] > 0)
        stitched[use] = scales[k] * exposures[k]["rate"][use]
    # fill any remaining gaps from the dimmest valid level
    for k in range(n):
        gap = np.isnan(stitched) & exposures[k]["valid"]
        stitched[gap] = scales[k] * exposures[k]["rate"][gap]
    return stitched, scales


# ==========================================================================
# 4) Run the whole thing and produce the diagnostic figure
# ==========================================================================
def run(E=8.0, pileup_pct=10.0, n_levels=4, bracket=10.0, t_s=1.0,
        sigma_px=5.0, orders=1, seed=0, intensities=None, path="stitch_demo.png"):
    rng = np.random.default_rng(seed)
    R_max = physics.true_rate_for_pileup(pileup_pct / 100.0, E)   # max usable rate (the cap)

    # intensity of each exposure relative to level 0 (whose peak sits at the cap).
    # Either an ideal geometric ladder (bracket**k), or a supplied list of REAL,
    # imperfect intensities -- e.g. from a mirror scan, where they only come out
    # approximately x10 because the attenuation curve is steep and the actuator is
    # not repeatable. Stitching does not need to know them; it fits them.
    levels = list(intensities) if intensities is not None else [bracket ** k for k in range(n_levels)]
    n_levels = len(levels)

    x = np.arange(320 - 40, 320 + 40 + orders * 40, 1.0)
    S = pattern_shape(x, orders=orders, sigma_px=sigma_px)
    true_ref = S * R_max                                          # reference-scale truth (level 0, noise-free)

    exposures = []
    for I in levels:
        rate, valid = record_exposure(S, R_max, I, t_s, rng)
        exposures.append(dict(rate=rate, valid=valid, counts=rate * t_s, intensity=I))

    stitched, scales = stitch(exposures)

    # brightest exposure that is VALID (unsaturated) at each pixel = best statistics
    kstar = np.zeros(len(x), dtype=int)
    for k in range(n_levels):
        kstar[exposures[k]["valid"]] = k              # ascending -> ends on brightest valid

    # "reported" region = where each exposure operates in its good decade, i.e. down
    # to the crossover (one bracket step above the cap). Below the crossover the last
    # exposure would run into the noise floor and is not counted as usable.
    crossover_counts = R_max * t_s / bracket
    level_arr = np.array(levels)
    exp_best_counts = true_ref * t_s * level_arr[kstar]     # expected counts, brightest valid level
    rep = np.isfinite(stitched) & (true_ref > 0) & (exp_best_counts >= crossover_counts)

    frac_err = (stitched[rep] - true_ref[rep]) / true_ref[rep]
    dyn_range = true_ref[rep].max() / true_ref[rep].min()
    # SEAM BIAS: bin away the Poisson noise and check the stitched curve tracks truth
    # with no systematic step at the joins -- the real fidelity test of the stitch.
    order = np.argsort(true_ref[rep])
    binned = np.array_split(order, min(12, max(1, rep.sum() // 8)))
    bias = [np.mean(stitched[rep][b]) / np.mean(true_ref[rep][b]) - 1 for b in binned if len(b)]
    max_seam_bias = float(np.max(np.abs(bias))) if bias else float("nan")

    # S/N uniformity: single-exposure (level 0 only) vs stitched, expected values
    sn_single = np.sqrt(true_ref * t_s)                     # if you only used level 0
    sn_stitch = np.sqrt(exp_best_counts)                    # brightest valid exposure
    sr_single = sn_single[rep].max() / sn_single[rep].min()
    sr_stitch = sn_stitch[rep].max() / sn_stitch[rep].min()

    if path:
        _plot(x, S, true_ref, exposures, stitched, scales, R_max, bracket,
              sn_single, sn_stitch, E, pileup_pct, t_s, path)

    return dict(scales=scales, ideal_scales=[1.0 / I for I in levels], intensities=levels,
                dyn_range=dyn_range, rms_frac_err=float(np.sqrt(np.mean(frac_err ** 2))),
                max_seam_bias=max_seam_bias,
                sn_range_single=sr_single, sn_range_stitched=sr_stitch,
                R_max=R_max, path=path)


def _plot(x, S, true_ref, exposures, stitched, scales, R_max, bracket,
          sn_single, sn_stitch, E, pileup_pct, t_s, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(exposures)))

    # (a) the raw bracketed exposures
    for k, (c, ex) in enumerate(zip(cols, exposures)):
        r = np.where(ex["valid"], ex["rate"], np.nan)
        ax[0, 0].semilogy(x, np.clip(r, 0.5, None), ".", ms=3, color=c,
                          label=f"level {k}  (x{bracket**k:.0f})")
    ax[0, 0].axhline(R_max, color="k", ls=":", lw=1, label="saturation cap")
    ax[0, 0].set_title("Four bracketed exposures (saturated points dropped)")
    ax[0, 0].set_xlabel("detector pixel"); ax[0, 0].set_ylabel("measured rate (cps)")
    ax[0, 0].legend(fontsize=7); ax[0, 0].grid(True, which="both", alpha=0.25)

    # (b) stitched vs true
    ax[0, 1].semilogy(x, np.clip(true_ref, 0.5, None), "-", color="#333", lw=1.6, label="true pattern")
    ax[0, 1].semilogy(x, np.clip(stitched, 0.5, None), ".", ms=4, color="#c0392b", label="stitched")
    ax[0, 1].set_title("Stitched reconstruction vs truth")
    ax[0, 1].set_xlabel("detector pixel"); ax[0, 1].set_ylabel("rate on reference scale (cps)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, which="both", alpha=0.25)

    # (c) S/N uniformity: single exposure vs stitched
    ax[1, 0].semilogy(x, np.clip(sn_single, 0.3, None), ".", ms=3, color="#e08a1e", label="single exposure (level 0)")
    ax[1, 0].semilogy(x, np.clip(sn_stitch, 0.3, None), ".", ms=3, color="#1f5fa8", label="stitched (4 exposures)")
    ax[1, 0].set_title("Signal-to-noise across the pattern")
    ax[1, 0].set_xlabel("detector pixel"); ax[1, 0].set_ylabel("S/N per pixel  (sqrt counts)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, which="both", alpha=0.25)

    # (d) fractional error of the stitched curve
    good = np.isfinite(stitched) & (true_ref > 0)
    ax[1, 1].plot(x[good], 100 * (stitched[good] - true_ref[good]) / true_ref[good], ".", ms=3, color="#2e8b57")
    ax[1, 1].axhline(0, color="k", lw=0.8)
    ax[1, 1].set_ylim(-30, 30)
    ax[1, 1].set_title("Stitched fractional error vs truth")
    ax[1, 1].set_xlabel("detector pixel"); ax[1, 1].set_ylabel("error (%)")
    ax[1, 1].grid(True, alpha=0.25)

    fitted = " ".join(f"{s:.3g}" for s in scales)
    fig.suptitle(f"Bracketing / stitching @ {E:g} keV  (cap={pileup_pct:g}% pileup -> "
                 f"{R_max:.2e} cps, {t_s:g}s/level; fitted cumulative scales: {fitted})",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    print("=== Single-Gaussian stitching (4 levels x10, 8 keV, 10% pileup cap) ===")
    r = run(orders=1)
    print(f"  fitted scales : {[round(s,4) for s in r['scales']]}")
    print(f"  ideal scales  : {[round(s,4) for s in r['ideal_scales']]}   (recovered blind from overlaps)")
    print(f"  recovered dynamic range (measured region): {r['dyn_range']:.1e}")
    print(f"  seam bias (systematic, Poisson binned out): {r['max_seam_bias']*100:.2f}%  <-- the fidelity of the stitch")
    print(f"  S/N range: single exposure {r['sn_range_single']:.0f}x  ->  stitched {r['sn_range_stitched']:.0f}x  (uniformity gain)")
    print(f"  wrote {r['path']}")
    print()
    print("=== Multi-order variant (4 orders, each 10x weaker) ===")
    r2 = run(orders=4, path="stitch_demo_orders.png")
    print(f"  seam bias {r2['max_seam_bias']*100:.2f}%; S/N range {r2['sn_range_single']:.0f}x -> {r2['sn_range_stitched']:.0f}x; wrote {r2['path']}")
    print()
    print("=== Intensities set by a REAL mirror scan (imperfect, non-repeatable) ===")
    mi = mirror_intensities(E=8.0, step_miscal=1.08, actuator_seed=1)
    print("  mirror angles (deg):     " + ", ".join(f"{t:.4f}" for t in mi["landed_theta"]))
    print("  realized intensities:    " + ", ".join(f"{I:.1f}" for I in mi["intensity"])
          + "   (ideal would be 1, 10, 100, 1000)")
    r3 = run(E=8.0, intensities=mi["intensity"], orders=1, path="stitch_mirror.png")
    print("  fitted scales (blind):   " + ", ".join(f"{s:.4g}" for s in r3["scales"]))
    print("  true 1/intensity:        " + ", ".join(f"{s:.4g}" for s in r3["ideal_scales"]))
    rec = max(abs(f / i - 1) for f, i in zip(r3["scales"][1:], r3["ideal_scales"][1:]))
    print(f"  scale recovery error:    {rec*100:.1f}%   (fit vs the true imperfect ratios)")
    print(f"  seam bias {r3['max_seam_bias']*100:.2f}%; S/N range {r3['sn_range_single']:.0f}x -> "
          f"{r3['sn_range_stitched']:.0f}x; wrote {r3['path']}")
