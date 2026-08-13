"""
run_sim.py -- the headline demonstration that ties all four pieces together.

Question we answer:  if the actuator has NO angle readout and is not repeatable,
can a detector-feedback loop still land on the correct mirror angle at every
energy?  We run the blind controller across 4-10 keV at a fixed 1% pileup target
and compare the angle it *found* (ground truth, revealed only for scoring) against
the angle the deterministic physics says it *should* be -- i.e. the operating
recipe from the report.  If they lie on top of each other, the loop has
reconstructed the (E -> theta) recipe without ever measuring an angle.

At each energy we also write a real-format AcquisitionNNNN/ at the locked point,
so the simulated output can be fed through the exact same analysis as real data.

Run:  python run_sim.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics
from actuator import PIM05Actuator
from detector import MythenDetector
from controller import auto_align, budget


# --------------------------------------------------------------------------
# The deterministic "right answer": solve pileup(theta) = target for theta.
# Pileup falls monotonically as the grazing angle grows (more angle -> less
# reflectivity -> less flux), so a simple bisection nails it.
# --------------------------------------------------------------------------
def ideal_angle(E, p_target, lo=0.10, hi=1.20):
    def pileup_at(theta):
        _, _, p = physics.forward(theta, E)
        return p
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pileup_at(mid) > p_target:      # too much flux -> need a larger angle
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Sweep the energy, run the blind loop at each, score against the recipe.
# --------------------------------------------------------------------------
def energy_sweep(energies=(4, 5, 6, 7, 8, 9, 10), p_target=0.01,
                 write_dir="sim_acquisitions"):
    os.makedirs(write_dir, exist_ok=True)
    rows = []
    for i, E in enumerate(energies):
        h = auto_align(E=E, p_target=p_target, theta0_deg=0.30, seed=1, verbose=False)
        found = h["theta"][-1]                 # angle the blind loop parked at (ground truth)
        want = ideal_angle(E, p_target)         # angle the recipe says it should be
        p_meas = h["p_meas"][-1]

        # write a real-format acquisition at the locked point, so it can be
        # analyzed exactly like a real run
        det = MythenDetector(energy_keV=E, seed=i)
        N_true, _, _ = physics.forward(found, E)
        det.acquire(beam_rate=N_true / det.peak_frac, n_frames=20, frame_s=1.0)
        det.write_acquisition(write_dir, acq_number=900 + int(E))

        rows.append((E, found, want, p_meas, h["converged"]))
        print(f"E={E:2d} keV | found {found:.4f} deg | recipe {want:.4f} deg | "
              f"err {(found-want)*1e3:+.2f} mdeg | pileup {p_meas*100:.2f}% | "
              f"{'ok' if h['converged'] else 'NO'}")
    return np.array([(r[0], r[1], r[2], r[3]) for r in rows])


# --------------------------------------------------------------------------
# Plot: blind-loop angle vs recipe curve, and the pileup it held at each energy.
# --------------------------------------------------------------------------
def plot(data, p_target=0.01, path="run_sim.png"):
    E, found, want, p = data.T
    # a smooth recipe curve for the background line
    Efine = np.linspace(E.min(), E.max(), 40)
    recipe = np.array([ideal_angle(e, p_target) for e in Efine])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.plot(Efine, recipe, "-", color="#c0392b", lw=1.5, label="recipe (deterministic)")
    ax1.plot(E, found, "o", ms=7, color="#1f5fa8", label="blind auto-align")
    ax1.set_xlabel("energy (keV)"); ax1.set_ylabel("mirror grazing angle (deg)")
    ax1.set_title(f"E -> theta recipe, reconstructed blind ({p_target*100:.0f}% pileup)")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(E, p * 100, "o-", color="#2e8b57")
    ax2.axhline(p_target * 100, color="#c0392b", ls="--", label="target")
    ax2.set_xlabel("energy (keV)"); ax2.set_ylabel("pileup held (%)")
    ax2.set_title("Pileup delivered at each energy")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Blind detector-feedback loop reproduces the operating recipe")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Budget accounting: because the actuator is non-repeatable, the cost to
# converge varies run to run.  We repeat each energy over several seeds and
# report mean +/- spread of the numbers controls actually needs:
#   - acquisitions (and total frames) to converge
#   - total commanded motor steps
#   - wall time = detector time + motor time
# --------------------------------------------------------------------------
def budget_sweep(energies=(4, 6, 8, 10), p_target=0.01, n_seeds=8,
                 n_frames=20, frame_s=0.2, warm_start=False, verbose=True):
    """Repeat each energy over several seeds and report mean +/- spread of the
       cost to converge.  warm_start=True carries ONE actuator across the whole
       ascending energy sweep (per seed), so each alignment begins where the
       previous energy left off instead of from a fixed cold start."""
    mode = "WARM start (carry position across energies)" if warm_start \
        else "COLD start (0.30 deg every time)"
    if verbose:
        print(f"\n=== Convergence budget -- {mode} "
              f"({n_seeds} seeds/energy, {n_frames} frames x {frame_s}s, PIM05 slew) ===")
        print(f"{'E':>5} | {'acqs':>10} | {'frames':>9} | {'steps':>12} | "
              f"{'meas s':>10} | {'motor s':>10} | {'wall s':>11}")
    per_E = {E: [] for E in energies}
    for s in range(n_seeds):
        # one actuator for the whole ascending sweep if warm-starting
        act = PIM05Actuator(theta0_deg=0.30, seed=s) if warm_start else None
        for E in energies:
            h = auto_align(E=E, p_target=p_target, theta0_deg=0.30,
                           n_frames=n_frames, frame_s=frame_s, seed=s,
                           act=(act if warm_start else None), verbose=False)
            b = budget(h)
            per_E[E].append([b["n_meas"], b["total_frames"], b["total_steps"],
                             b["meas_time_s"], b["motor_time_s"], b["wall_time_s"]])
    out = []
    for E in energies:
        recs = np.array(per_E[E], float)
        m, sd = recs.mean(0), recs.std(0)
        if verbose:
            print(f"{E:5.1f} | {m[0]:4.1f}+/-{sd[0]:3.1f} | {m[1]:9.0f} | "
                  f"{m[2]:6.0f}+/-{sd[2]:4.0f} | {m[3]:5.1f}+/-{sd[3]:3.1f} | "
                  f"{m[4]:5.1f}+/-{sd[4]:3.1f} | {m[5]:6.1f}+/-{sd[5]:4.1f}")
        out.append((E, m, sd))
    return out


def plot_budget(cold, warm, path="run_sim_budget.png"):
    """Compare cold vs warm start: wall time (detector+motor) and total steps."""
    E = np.array([r[0] for r in cold])
    def col(rows, i): return np.array([r[1][i] for r in rows])
    def col_sd(rows, i): return np.array([r[2][i] for r in rows])
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # panel 1: stacked detector+motor time, cold bars vs warm bars side by side
    for k, (rows, off, hatch, lab) in enumerate([(cold, -w/2, "", "cold"),
                                                 (warm, +w/2, "//", "warm")]):
        meas, motor = col(rows, 3), col(rows, 4)
        ax1.bar(E + off, meas, width=w, color="#1f5fa8", hatch=hatch,
                edgecolor="white", label=f"detector ({lab})")
        ax1.bar(E + off, motor, width=w, bottom=meas, color="#e08a1e", hatch=hatch,
                edgecolor="white", yerr=col_sd(rows, 5), capsize=3,
                error_kw=dict(ecolor="#444"), label=f"motor ({lab})")
    ax1.set_xlabel("energy (keV)"); ax1.set_ylabel("wall time to converge (s)")
    ax1.set_title("Time budget: cold vs warm start"); ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=7, ncol=2)

    # panel 2: total commanded steps, cold vs warm
    ax2.errorbar(E, col(cold, 2), yerr=col_sd(cold, 2), fmt="o-", color="#8e44ad",
                 capsize=4, label="cold start")
    ax2.errorbar(E, col(warm, 2), yerr=col_sd(warm, 2), fmt="s--", color="#2e8b57",
                 capsize=4, label="warm start")
    ax2.set_xlabel("energy (keV)"); ax2.set_ylabel("total commanded steps")
    ax2.set_title("Motor step budget: cold vs warm"); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Convergence cost, PIM05 (mean +/- spread over seeds)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


if __name__ == "__main__":
    print("=== Energy sweep: blind loop vs deterministic recipe, 1% pileup ===")
    data = energy_sweep(p_target=0.01)
    rms = np.sqrt(np.mean((data[:, 1] - data[:, 2]) ** 2)) * 1e3
    print(f"\nRMS(angle found - recipe) = {rms:.2f} mdeg across the band")
    print(f"wrote plot to {plot(data, 0.01, 'run_sim.png')}")

    # coarse sweep, cold vs warm start
    cold = budget_sweep(p_target=0.01, warm_start=False)
    warm = budget_sweep(p_target=0.01, warm_start=True)
    tot_c = sum(r[1][4] for r in cold); tot_w = sum(r[1][4] for r in warm)
    print(f"\nCoarse 4-10 keV sweep motor time: cold {tot_c:.1f}s -> warm {tot_w:.1f}s. "
          f"(4 keV is the first point so it can't be warm-started; the middle energies "
          f"drop sharply, e.g. 6 keV {cold[1][1][4]:.1f}s -> {warm[1][1][4]:.1f}s.)")
    print(f"wrote plot to {plot_budget(cold, warm)}")

    # fine 0.1 keV scan -- the real use case, where consecutive angles are close
    # and warm-starting from the previous point is a big win.
    fine = tuple(round(8.0 + 0.1 * k, 1) for k in range(11))     # 8.0 .. 9.0 keV
    fcold = budget_sweep(energies=fine, warm_start=False, n_seeds=4, verbose=False)
    fwarm = budget_sweep(energies=fine, warm_start=True, n_seeds=4, verbose=False)
    # average per-point motor time, skipping the first point (no warm history yet)
    mc = np.mean([r[1][4] for r in fcold[1:]])
    mw = np.mean([r[1][4] for r in fwarm[1:]])
    print(f"Fine 0.1 keV scan (8-9 keV): warm start cuts per-point motor time "
          f"{mc:.2f}s -> {mw:.2f}s ({(1 - mw / mc) * 100:.0f}%).")
