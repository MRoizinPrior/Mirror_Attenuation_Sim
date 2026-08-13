"""
detector.py -- simulated DECTRIS MYTHEN2 (4 mm) 1-D strip detector.

Turns a *true* photon rate (from physics x actuator) into what the real detector
actually writes out:

  * 640 strips, one column of counts per strip
  * paralyzable pileup at the strip level  (M = N exp(-N tau))
  * Poisson counting noise, integrated over a frame of length `frame_s`
  * a run = `n_frames` frames  (frame length and count are the requested variables)
  * output written in the REAL file format: a folder AcquisitionNNNN/ containing
    AcquisitionNNNN.cfg (XML) and Frame0001.dat ... with 640 "channel count" rows
  * the built-in DECTRIS rate (deadtime) correction, invertible below the turnover

The controller in Piece 4 will only ever get numbers back from `acquire()` /
`beam_rate()` -- exactly what it would read from the real DAQ.

Deps: numpy, scipy, matplotlib.  Uses physics.py for tau(E).
"""

import os
import numpy as np
from scipy.special import lambertw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics


class MythenDetector:
    def __init__(self, energy_keV=8.0, threshold_keV=5.4,
                 n_channels=640, beam_center=453, beam_sigma_ch=0.6,
                 background_cps=2.0, seed=None):
        self.E = energy_keV
        self.threshold_keV = threshold_keV
        self.nch = n_channels
        self.tau = physics.tau_seconds(energy_keV)      # deadtime for this energy/threshold
        self.beam_center = beam_center
        self.beam_sigma = beam_sigma_ch
        self.background_cps = background_cps              # flat scatter floor, absolute cps/strip
        self.rng = np.random.default_rng(seed)

        # Normalized beam profile over the strips: a narrow Gaussian (~1-2 strips wide,
        # like the real ~50-100 um beam on a 50 um pitch).  Normalized to sum to 1 so
        # that (total beam rate) x profile gives the true rate on each strip.  The
        # scatter floor is added separately in acquire(), so it does NOT steal flux
        # from the beam.
        ch = np.arange(n_channels)
        prof = np.exp(-0.5 * ((ch - beam_center) / beam_sigma_ch) ** 2)
        self.profile = prof / prof.sum()                 # sums to 1 over all strips
        self.peak_frac = self.profile.max()              # fraction of the beam on the brightest strip

    # ------------------------------------------------------------------
    # Acquire a run.  `beam_rate` is the TRUE total photon rate into the beam
    # (ph/s); it may be a scalar, or a function beam_rate(frame_index) so a
    # driver can inject drift frame-by-frame.
    # ------------------------------------------------------------------
    def acquire(self, beam_rate, n_frames=100, frame_s=1.0):
        frames = np.zeros((n_frames, self.nch), dtype=np.int64)
        rates_used = np.zeros(n_frames)
        for i in range(n_frames):
            N_tot = beam_rate(i) if callable(beam_rate) else beam_rate
            rates_used[i] = N_tot
            N_ch = N_tot * self.profile + self.background_cps  # true rate per strip [ph/s] (+ scatter floor)
            M_ch = N_ch * np.exp(-N_ch * self.tau)            # piled-up (measured) rate per strip
            frames[i] = self.rng.poisson(M_ch * frame_s)      # integer counts this frame
        self.frames = frames
        self.frame_s = frame_s
        self.n_frames = n_frames
        self.rates_used = rates_used
        return frames

    # ------------------------------------------------------------------
    # Acquire from an ARBITRARY per-strip true-rate map (len = n_channels).
    # Used for the sample scan, where the "beam" on the detector is a whole
    # diffraction pattern (many orders on different strips) rather than one spot.
    # Same physics as acquire(): per-strip pileup + Poisson noise per frame.
    # ------------------------------------------------------------------
    def acquire_pattern(self, rate_per_strip, n_frames=20, frame_s=1.0):
        rate_per_strip = np.asarray(rate_per_strip, dtype=float)
        frames = np.zeros((n_frames, self.nch), dtype=np.int64)
        for i in range(n_frames):
            N_ch = rate_per_strip + self.background_cps       # true rate per strip [ph/s]
            M_ch = N_ch * np.exp(-N_ch * self.tau)            # per-strip paralyzable pileup
            frames[i] = self.rng.poisson(M_ch * frame_s)      # integer counts this frame
        self.frames = frames
        self.frame_s = frame_s
        self.n_frames = n_frames
        self.rates_used = np.full(n_frames, rate_per_strip.sum())
        return frames

    # ------------------------------------------------------------------
    # DECTRIS deadtime (rate) correction: invert M = N exp(-N tau) for N,
    # taking the physical branch below the turnover N < 1/tau.
    # ------------------------------------------------------------------
    def correct_rate(self, measured_rate):
        x = -measured_rate * self.tau
        x = np.clip(x, -1.0 / np.e + 1e-12, 0.0)              # keep on the principal branch
        return np.real(-lambertw(x, 0)) / self.tau

    # ------------------------------------------------------------------
    # What the controller / analysis actually reads: the beam signal.
    # Returns measured & deadtime-corrected rate on the beam, per frame,
    # summed over a small window of strips around the beam center.
    # ------------------------------------------------------------------
    def beam_counts_per_frame(self, half_window=3):
        lo, hi = self.beam_center - half_window, self.beam_center + half_window + 1
        return self.frames[:, lo:hi].sum(axis=1)             # counts per frame in the beam window

    def beam_rate(self, half_window=4, corrected=True):
        """Mean integrated beam rate over the run (ph/s) and its 1-sigma uncertainty.
           Pileup is a per-strip nonlinearity, so we deadtime-correct EACH strip and
           then sum over the beam window -- not the other way around."""
        lo, hi = self.beam_center - half_window, self.beam_center + half_window + 1
        strip_rate = self.frames[:, lo:hi] / self.frame_s        # measured cps, per frame per strip
        if corrected:
            strip_rate = self.correct_rate(strip_rate)           # invert deadtime per strip
        per_frame = strip_rate.sum(axis=1)                       # integrate over the window
        return per_frame.mean(), per_frame.std(ddof=1) / np.sqrt(len(per_frame))

    def measured_pileup(self):
        """Pileup fraction on the brightest strip, ESTIMATED FROM THE DATA.
           This is what the controller in Piece 4 will servo on -- no ground-truth peeking."""
        summed = self.frames.sum(axis=0)                         # counts per strip, whole run
        peak = summed.max() / (self.n_frames * self.frame_s)     # measured peak-strip rate [cps]
        N_peak = self.correct_rate(peak)                         # true peak-strip rate
        return 1.0 - np.exp(-N_peak * self.tau)

    # ------------------------------------------------------------------
    # Write the run out in the REAL MYTHEN file format.
    # ------------------------------------------------------------------
    def write_acquisition(self, out_dir, acq_number=63, rate_correction=False):
        folder = os.path.join(out_dir, f"Acquisition{acq_number:04d}")
        os.makedirs(folder, exist_ok=True)
        cfg = f"""<config>
  <version>3.0</version>
  <comment>SIMULATED</comment>
  <acquisition>{acq_number}</acquisition>
  <frames>{self.n_frames}</frames>
  <exposureTime units="ms">{self.frame_s*1000:.0f}</exposureTime>
  <energy units="keV">{self.E}</energy>
  <channels>{self.nch}</channels>
  <flatfieldCorrection>on</flatfieldCorrection>
  <rateCorrection>{'on' if rate_correction else 'off'}</rateCorrection>
  <modules><module>
      <nChannels>{self.nch}</nChannels>
      <threshold units="keV">{self.threshold_keV}</threshold>
      <material>Si</material><thickness units="um">320</thickness><width units="um">4000</width>
      <tau units="ns">{self.tau*1e9:.0f}</tau>
  </module></modules>
</config>
"""
        with open(os.path.join(folder, f"Acquisition{acq_number:04d}.cfg"), "w") as f:
            f.write(cfg)
        ch = np.arange(self.nch)
        for i in range(self.n_frames):
            np.savetxt(os.path.join(folder, f"Frame{i+1:04d}.dat"),
                       np.column_stack([ch, self.frames[i]]), fmt="%d")
        return folder

    # ------------------------------------------------------------------
    # Display: summed spectrum + beam counts per frame (like the report's Fig 19).
    # ------------------------------------------------------------------
    def plot(self, path="detector_sim.png", half_window=3):
        summed = self.frames.sum(axis=0)
        cpf = self.beam_counts_per_frame(half_window)
        t = np.arange(self.n_frames) * self.frame_s
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.semilogy(np.arange(self.nch), np.maximum(summed, 0.5), color="#c0392b", lw=0.9)
        ax1.set_xlim(self.beam_center - 30, self.beam_center + 30)
        ax1.set_xlabel("strip (channel)"); ax1.set_ylabel(f"counts (sum of {self.n_frames} frames)")
        ax1.set_title("Summed spectrum"); ax1.grid(True, which="both", alpha=0.3)
        ax2.plot(t, cpf / self.frame_s, "o", ms=3, color="#1f5fa8")
        ax2.axhline((cpf / self.frame_s).mean(), color="#1f5fa8", lw=1)
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("beam rate (measured cps)")
        ax2.set_title(f"Beam signal per frame ({self.frame_s*1000:.0f} ms x {self.n_frames})")
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"Simulated MYTHEN2 @ {self.E} keV  (tau={self.tau*1e9:.0f} ns)")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        return path


# --------------------------------------------------------------------------
# Self-test: acquire a run at the 1%-pileup rate, write files, correct, plot.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    E = 8.0
    det = MythenDetector(energy_keV=E, seed=0)
    # We want the BRIGHTEST strip to sit at 1% pileup.  physics gives the per-strip
    # rate for 1% pileup; scale up to a TOTAL beam rate since the beam spreads over
    # a couple of strips (only peak_frac of it lands on the brightest one).
    strip_target = physics.true_rate_for_pileup(0.01, E)
    N_total = strip_target / det.peak_frac
    print(f"Target: 1% pileup on peak strip ({strip_target:.3e} ph/s/strip)"
          f" -> total beam {N_total:.3e} ph/s")

    det.acquire(beam_rate=N_total, n_frames=100, frame_s=1.0)      # <-- frame length & count
    meas, err_m = det.beam_rate(corrected=False)
    corr, err_c = det.beam_rate(corrected=True)
    print(f"  measured integrated rate = {meas:.3e} +/- {err_m:.1e} ph/s")
    print(f"  deadtime-corrected       = {corr:.3e} +/- {err_c:.1e} ph/s  (truth {N_total:.3e})")
    print(f"  pileup measured from data = {det.measured_pileup()*100:.2f}%  (target 1.00%)")

    folder = det.write_acquisition(out_dir=".", acq_number=900, rate_correction=False)
    print(f"  wrote real-format data to {folder}")
    print(f"  wrote plot to {det.plot('detector_sim.png')}")
