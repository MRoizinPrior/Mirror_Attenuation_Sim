"""
controller.py -- closed-loop auto-alignment of the mirror attenuator.

This is the whole point of the exercise.  The actuator is NOT repeatable, so we
cannot open-loop it to a known angle.  Instead we use the detector as a flux
sensor and servo the mirror until the measured pileup matches a target we can set
anywhere from 1% to 10%.

Hard rule that mirrors the real hardware:
    the controller may ONLY call  actuator.step(n)      (command relative steps)
    and read                      the detector          (counts -> pileup / rate)
    it may NEVER read actuator.theta_deg (there is no angle readout on the real rig).

We keep a `sense()` helper that DOES look at the true angle -- but only to emulate
the physical world (true angle -> true flux -> detector counts).  The control
logic below never touches that ground truth; it only uses `sense()`'s returned
measurement.  The true angle is recorded purely so we can plot how close we got.

Strategy:
  Phase A  BRACKET  -- step in the flux-reducing direction with a geometrically
                       growing step until the pileup crosses the target (sign flip
                       of the error).  This spans the large dynamic range cheaply.
  Phase B  SECANT   -- with two measurements we estimate the local slope
                       d(ln rate)/d(step) and take Newton steps to the target,
                       re-estimating the slope each iteration (so it adapts and
                       naturally absorbs backlash / the +/-20% step jitter).
  Phase C  HOLD     -- optional: sit and keep measuring; thermal drift walks the
                       pileup off target, and the loop trims it back.  Shows the
                       closed loop beats the drift the open loop could not.

Deps: numpy, matplotlib, and our physics/actuator/detector modules.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics
from actuator import PIM05Actuator
from detector import MythenDetector


# --------------------------------------------------------------------------
# The sensor.  Couples the (hidden) true angle -> physics -> detector, and hands
# back ONLY what the real DAQ would give: a measured pileup and corrected rate.
# The true angle/pileup are returned too, but tagged as ground truth for plotting.
# --------------------------------------------------------------------------
def sense(act, det, E, n_frames, frame_s):
    theta = act.theta_deg                                  # GROUND TRUTH -- harness only
    N_true, _, p_true = physics.forward(theta, E)          # true per-strip rate & pileup
    det.acquire(beam_rate=N_true / det.peak_frac,          # spread beam so peak strip carries N_true
                n_frames=n_frames, frame_s=frame_s)
    act.advance_time(n_frames * frame_s)                   # clock runs during the exposure -> drift
    p_meas = det.measured_pileup()                         # <-- the only thing the controller trusts
    N_meas = -np.log(1.0 - p_meas) / det.tau               # equivalent measured rate on the peak strip
    return p_meas, N_meas, p_true, theta


# --------------------------------------------------------------------------
# The auto-aligner.
# --------------------------------------------------------------------------
def auto_align(E=8.0, p_target=0.01, theta0_deg=0.30,
               n_frames=20, frame_s=0.2, tol_rel=0.02,
               bracket_step0=200, max_secant=25, seed=0, act=None,
               beam_sigma_ch=0.6, verbose=True):
    """Drive the mirror until measured pileup == p_target (1-10%).
       Returns a history dict for plotting.

       act: pass an existing PIM05Actuator to WARM-START from its current
       position (e.g. carry it across an energy sweep); leave None for a fresh
       cold start at theta0_deg."""
    if act is None:
        act = PIM05Actuator(theta0_deg=theta0_deg, seed=seed)   # cold start
    steps0 = act.total_steps                                    # so we can bill THIS alignment only
    det = MythenDetector(energy_keV=E, seed=seed, beam_sigma_ch=beam_sigma_ch)

    # The controller is allowed to know the detector calibration (tau, the pileup
    # model) -- that is just data-analysis, not a position readout.  So it can turn
    # the target pileup into a target *rate* on the peak strip.
    N_target = physics.true_rate_for_pileup(p_target, E)
    ln_target = np.log(N_target)

    steps_cum = 0                                          # our only position proxy: commanded steps
    hist = {"step": [], "p_meas": [], "p_true": [], "theta": [], "N_meas": [], "phase": []}

    def record(phase, p_meas, N_meas, p_true, theta):
        hist["step"].append(steps_cum); hist["p_meas"].append(p_meas)
        hist["N_meas"].append(N_meas); hist["p_true"].append(p_true)
        hist["theta"].append(theta); hist["phase"].append(phase)

    # error in log-rate: e>0 means too much flux -> need a LARGER angle -> +steps
    def err(N_meas):
        return np.log(N_meas) - ln_target

    # -------- initial measurement --------
    p, Nm, pt, th = sense(act, det, E, n_frames, frame_s)
    record("start", p, Nm, pt, th)
    e = err(Nm)
    if verbose:
        print(f"start: pileup {p*100:5.2f}% (true {pt*100:5.2f}%)  angle {th:.4f} deg  e={e:+.3f}")

    # ---------------- Phase A: BRACKET ----------------
    # +steps increase the grazing angle -> less reflectivity -> less flux -> lower
    # pileup.  So to correct an error of sign(e) we command steps of sign(e).
    # Grow the step geometrically until the error flips sign (we've straddled target).
    if abs(e) > tol_rel:
        inc = bracket_step0
        while True:
            d = int(np.sign(e) * inc)
            act.step(d); steps_cum += d
            p, Nm, pt, th = sense(act, det, E, n_frames, frame_s)
            record("bracket", p, Nm, pt, th)
            e_new = err(Nm)
            if verbose:
                print(f"brkt : step {d:+6d} -> pileup {p*100:5.2f}%  angle {th:.4f} deg  e={e_new:+.3f}")
            if np.sign(e_new) != np.sign(e) or abs(e_new) <= tol_rel:
                e = e_new
                break                                     # bracketed (or already good enough)
            e = e_new
            inc *= 2                                       # widen the search each time

    # ---------------- Phase B: SECANT ----------------
    # Use the last two recorded points to estimate the local sensitivity
    # slope = d(e)/d(step), then take a damped Newton step  delta = -e/slope.
    for _ in range(max_secant):
        if abs(e) <= tol_rel:
            break
        s0, s1 = hist["step"][-2], hist["step"][-1]
        e0 = err(hist["N_meas"][-2]); e1 = err(hist["N_meas"][-1])
        slope = (e1 - e0) / (s1 - s0) if s1 != s0 else 0.0
        # slope should be negative (more steps -> less flux).  If noise makes it
        # non-negative, fall back to a modest step in the correcting direction.
        if slope >= -1e-6:
            d = int(np.sign(e) * 100)
        else:
            d = int(np.clip(-e / slope, -4000, 4000))      # Newton step, clipped for safety
            if d == 0:
                d = int(np.sign(e))                        # never stall exactly at 0
        act.step(d); steps_cum += d
        p, Nm, pt, th = sense(act, det, E, n_frames, frame_s)
        record("secant", p, Nm, pt, th)
        e = err(Nm)
        if verbose:
            print(f"scnt : step {d:+6d} -> pileup {p*100:5.2f}%  angle {th:.4f} deg  e={e:+.3f}")

    converged = abs(e) <= tol_rel
    if verbose:
        print(f"{'CONVERGED' if converged else 'stopped'}: "
              f"pileup {hist['p_meas'][-1]*100:.2f}% (target {p_target*100:.2f}%), "
              f"true angle {hist['theta'][-1]:.4f} deg, {steps_cum:+d} net steps\n")
    hist["converged"] = converged
    hist["p_target"] = p_target
    hist["N_target"] = N_target
    hist["E"] = E
    hist["_act"] = act; hist["_det"] = det                 # keep alive for an optional hold phase
    hist["_cfg"] = dict(n_frames=n_frames, frame_s=frame_s, tol_rel=tol_rel, ln_target=ln_target)
    hist["_steps_cum"] = steps_cum
    # --- budget bookkeeping (what controls needs to size the loop) ---
    hist["n_meas"] = len(hist["p_meas"])                   # detector acquisitions to converge
    hist["total_steps"] = act.total_steps - steps0         # commanded steps for THIS alignment
    hist["net_steps"] = steps_cum                          # net displacement from start
    hist["step_rate_hz"] = act.step_rate_hz                # PIM05 slew, for the time budget
    return hist


# --------------------------------------------------------------------------
# Convert a converged run into a concrete time/step budget for controls.
# Wall time = detector time + motor time.  Detector time is n_frames x frame_s
# per acquisition plus a small per-acquisition arm/handshake overhead.  Motor
# time is total commanded steps divided by the picomotor step rate.
# --------------------------------------------------------------------------
def budget(hist, step_rate_hz=None, acq_overhead_s=0.05):
    cfg = hist["_cfg"]
    if step_rate_hz is None:                                # default to the actuator's own slew
        step_rate_hz = hist.get("step_rate_hz", 1667.0)
    total_frames = hist["n_meas"] * cfg["n_frames"]
    meas_time = total_frames * cfg["frame_s"] + hist["n_meas"] * acq_overhead_s
    motor_time = hist["total_steps"] / step_rate_hz
    return dict(n_meas=hist["n_meas"], total_frames=total_frames,
                total_steps=hist["total_steps"], net_steps=hist["net_steps"],
                meas_time_s=meas_time, motor_time_s=motor_time,
                wall_time_s=meas_time + motor_time)


# --------------------------------------------------------------------------
# Phase C (optional): hold and let drift fight the loop.  Every `dwell_s` we
# re-measure; if we've walked outside tolerance, take one corrective step batch.
# --------------------------------------------------------------------------
def hold_and_track(hist, duration_s=600, dwell_s=30, trim_gain=0.8, verbose=True):
    act, det = hist["_act"], hist["_det"]
    cfg = hist["_cfg"]; E = hist["E"]; ln_target = cfg["ln_target"]
    steps_cum = hist["_steps_cum"]
    track = {"t": [], "p_meas": [], "p_true": [], "theta": [], "corrected": []}
    t = 0.0
    # rough local sensitivity (steps per unit log-rate) from the converged secant leg
    s0, s1 = hist["step"][-2], hist["step"][-1]
    e0 = np.log(hist["N_meas"][-2]) - ln_target
    e1 = np.log(hist["N_meas"][-1]) - ln_target
    slope = (e1 - e0) / (s1 - s0) if (s1 != s0 and e1 != e0) else -1.0
    if slope >= -1e-9:
        slope = -1.0
    while t < duration_s:
        act.advance_time(dwell_s - cfg["n_frames"] * cfg["frame_s"])   # idle between checks
        p, Nm, pt, th = sense(act, det, E, cfg["n_frames"], cfg["frame_s"])
        e = np.log(Nm) - ln_target
        corrected = False
        if abs(e) > cfg["tol_rel"]:                        # drifted out -> trim back
            d = int(np.clip(trim_gain * (-e / slope), -2000, 2000))
            if d != 0:
                act.step(d); steps_cum += d
                p, Nm, pt, th = sense(act, det, E, cfg["n_frames"], cfg["frame_s"])
                corrected = True
        t += dwell_s
        track["t"].append(t); track["p_meas"].append(p); track["p_true"].append(pt)
        track["theta"].append(th); track["corrected"].append(corrected)
        if verbose:
            flag = " <trim" if corrected else ""
            print(f"hold t={t:4.0f}s: pileup {p*100:5.2f}%  angle {th:.4f} deg{flag}")
    return track


# --------------------------------------------------------------------------
# Plot: convergence to target, plus (if run) the hold/drift-tracking phase.
# --------------------------------------------------------------------------
def plot(hist, track=None, path="controller_sim.png"):
    it = np.arange(len(hist["p_meas"]))
    fig, axes = plt.subplots(1, 2 if track is None else 3,
                             figsize=(11 if track is None else 15, 4.3))
    axes = np.atleast_1d(axes)

    # panel 1: pileup vs iteration
    ax = axes[0]
    ax.semilogy(it, np.array(hist["p_meas"]) * 100, "o-", color="#1f5fa8", label="measured")
    ax.axhline(hist["p_target"] * 100, color="#c0392b", ls="--", label="target")
    ax.set_xlabel("control iteration"); ax.set_ylabel("pileup (%)")
    ax.set_title("Convergence to target pileup"); ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # panel 2: the (hidden) true angle the loop actually parked at
    ax = axes[1]
    ax.plot(it, hist["theta"], "s-", color="#2e8b57")
    ax.set_xlabel("control iteration"); ax.set_ylabel("true mirror angle (deg)")
    ax.set_title(f"Angle found (blind)  final {hist['theta'][-1]:.4f} deg")
    ax.grid(True, alpha=0.3)

    # panel 3 (optional): hold phase, pileup vs time with trims marked
    if track is not None:
        ax = axes[2]
        t = np.array(track["t"]); p = np.array(track["p_meas"]) * 100
        ax.plot(t, p, "-o", ms=3, color="#8e44ad")
        corr = np.array(track["corrected"])
        if corr.any():
            ax.plot(t[corr], p[corr], "v", color="#c0392b", label="trim applied")
            ax.legend(loc="upper right", fontsize=8)
        ax.axhline(hist["p_target"] * 100, color="#c0392b", ls="--")
        ax.set_xlabel("time (s)"); ax.set_ylabel("pileup (%)")
        ax.set_title("Hold: drift vs loop"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Mirror auto-align @ {hist['E']} keV  (target {hist['p_target']*100:.0f}% pileup)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Self-test: (1) hit three different pileup targets from the same flooded start,
#            (2) run a hold phase on the 1% case to show drift tracking.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Variable pileup spec: align to 1%, 5%, 10% from a flooded start ===")
    for pt in (0.01, 0.05, 0.10):
        h = auto_align(E=8.0, p_target=pt, theta0_deg=0.30, seed=1)

    print("=== Hold phase: lock at 1%, then let 600 s of thermal drift fight the loop ===")
    h = auto_align(E=8.0, p_target=0.01, theta0_deg=0.30, seed=1, verbose=False)
    print(f"locked at {h['p_meas'][-1]*100:.2f}% (true angle {h['theta'][-1]:.4f} deg)")
    track = hold_and_track(h, duration_s=600, dwell_s=30)
    print(f"wrote plot to {plot(h, track, 'controller_sim.png')}")
