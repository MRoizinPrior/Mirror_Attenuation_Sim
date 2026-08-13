"""
acquisition.py -- the full automated data-collection run.

This ties the mirror auto-align, the sample stage, and the detector into one
recipe, exactly as it would run on the beamline:

  for each energy E = 4..10 keV:
      1. move the sample OUT of the beam (rotate horizontal + lift 2 mm)
      2. auto-align the mirror to the target pileup on the direct beam
         (this calibrates the delivered flux at this energy)
      3. move the sample back IN
      4. for each grazing angle 0.2 .. 0.7 deg:
             rotate the sample and acquire a spectrum
      5. plot the six spectra for this energy

Every mechanical move is logged, so the printout reads like an acquisition log.
Output: one figure per energy (six sub-plots, MYTHEN counts vs strip), plus the log.

Deps: numpy, matplotlib, physics/controller/detector/sample.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics
import controller
import detector
import sample

GRAZING_ANGLES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)     # sample incidence angles to scan
ENERGIES = (4, 5, 6, 7, 8, 9, 10)                   # keV


def run_full_sequence(E_min=4.0, E_max=10.0, n_energies=7,
                      alpha_min=0.2, alpha_max=0.7, n_angles=6,
                      energies=None, grazing=None, p_target=0.01,
                      n_frames=20, frame_s=1.0, seed=1, out_prefix="sample_scan",
                      slit_um=100.0, L_m=0.698,
                      save_files=False, save_dir="scan_acquisitions", verbose=True):
    """Run the whole out/calibrate/in/scan recipe. Returns (figure_paths, log).

    Energy scan  : n_energies points evenly from E_min to E_max keV.
    Angle scan   : n_angles points evenly from alpha_min to alpha_max deg.
    (Pass explicit `energies`/`grazing` arrays to override the even spacing.)

    save_files : also write every spectrum in the real MYTHEN Acquisition/Frame
                 format, one folder per (energy, angle), into save_dir."""
    if energies is None:
        energies = np.linspace(E_min, E_max, int(n_energies))
    if grazing is None:
        grazing = np.linspace(alpha_min, alpha_max, int(n_angles))

    # adaptive sub-plot grid for however many angles were requested
    n = len(grazing)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    stage = sample.SampleStage()
    paths = []
    summary = {}          # {E: (grazing_angles, integrated_rates)} for the rocking curve

    for E in energies:
        if verbose:
            stage.log.append(f"===== ENERGY {E:.2f} keV =====")

        # slit height -> beam width on the detector (strips) at this energy
        sigma = physics.beam_sigma_strips(slit_um, E, L_m)

        # 1) sample OUT, 2) calibrate the mirror to target pileup on the direct beam
        stage.move_out()
        h = controller.auto_align(E=E, p_target=p_target, n_frames=n_frames,
                                  frame_s=frame_s, seed=seed, beam_sigma_ch=sigma,
                                  verbose=False)
        theta = h["theta"][-1]
        # 3) sample IN, 4) scan the grazing angles
        stage.move_in()
        det = detector.MythenDetector(energy_keV=E, seed=seed, beam_sigma_ch=sigma)
        # delivered TOTAL flux: peak strip sits at the pileup target, so the total
        # (spread over the beam width) is larger for a wider beam -> flux ~ slit
        N_inc = physics.forward(theta, E)[0] / det.peak_frac
        stage.log.append(f"   mirror calibrated: {h['p_meas'][-1]*100:.2f}% pileup "
                         f"(angle {theta:.3f} deg); slit {slit_um:.0f} um -> beam "
                         f"{physics.beam_fwhm_um(slit_um, E, L_m):.0f} um, flux {N_inc:.2e} ph/s")
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows),
                                 squeeze=False)
        axes = axes.ravel()
        integrated = []
        for j, alpha in enumerate(grazing):
            stage.rotate_to(alpha)
            pattern = sample.diffraction_pattern(E, alpha, N_inc, det,
                                                 peak_sigma_strip=sigma)
            det.acquire_pattern(pattern, n_frames=n_frames, frame_s=frame_s)
            summed = det.frames.sum(axis=0)
            R = sample._sample_reflectivity(E, alpha)

            # integrated (background-subtracted) diffracted rate -> the rocking curve
            bg_counts = det.background_cps * det.nch * frame_s * n_frames
            signal = max(det.frames.sum() - bg_counts, 1e-9)
            integrated.append(signal / (n_frames * frame_s))

            # optionally save this spectrum in the real MYTHEN format.
            # acq number encodes energy and angle: round(E*10)*1000 + round(alpha*100)
            if save_files:
                acq = int(round(E * 10)) * 1000 + int(round(alpha * 100))
                det.write_acquisition(out_dir=save_dir, acq_number=acq)

            ax = axes[j]
            ax.semilogy(np.arange(det.nch), np.maximum(summed, 0.5),
                        color="#c0392b", lw=0.9)
            ax.set_xlim(390, 639)         # show the full upper half of the detector
            ax.set_ylim(bottom=0.5)
            ax.set_title(f"grazing {alpha:.2f} deg   (R = {R*100:.2g}%)", fontsize=10)
            ax.set_xlabel("detector strip"); ax.set_ylabel("counts")
            ax.grid(True, which="both", alpha=0.25)

        for k in range(n, len(axes)):     # hide any unused sub-plots
            axes[k].axis("off")

        fig.suptitle(f"Sample diffraction spectra @ {E:.2f} keV   "
                     f"(mirror held at {p_target*100:.0f}% pileup calibration, "
                     f"{n_frames}x{frame_s:g}s frames)", fontsize=12)
        fig.tight_layout()
        path = f"{out_prefix}_{E:04.1f}keV.png"
        fig.savefig(path, dpi=130); plt.close(fig)
        paths.append(path)
        summary[E] = (np.array(grazing, float), np.array(integrated))
        if verbose:
            note = f" (+ real files in {save_dir}/)" if save_files else ""
            print(f"E={E:5.2f} keV: calibrated + scanned {len(grazing)} angles -> {path}{note}")

    # summary rocking / reflectivity curve across all energies
    summary_path = plot_scan_summary(summary, p_target, path=f"{out_prefix}_summary.png")
    paths.append(summary_path)
    return paths, stage.log


def plot_scan_summary(summary, p_target=0.01, path="sample_scan_summary.png"):
    """Integrated diffracted intensity vs grazing angle, one curve per energy.
       This is effectively the sample's reflectivity / rocking curve: it falls
       steeply as the grazing angle passes the (energy-dependent) Si critical angle."""
    fig, ax = plt.subplots(figsize=(8, 5))
    energies = sorted(summary)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(energies)))
    for c, E in zip(cmap, energies):
        angles, inten = summary[E]
        ax.semilogy(angles, inten, "o-", color=c, label=f"{E:.1f} keV")
    ax.set_xlabel("sample grazing angle (deg)")
    ax.set_ylabel("integrated diffracted rate (counts/s)")
    ax.set_title(f"Rocking curve: integrated intensity vs grazing angle\n"
                 f"(mirror held at {p_target*100:.0f}% pileup; sample = 200 l/mm bare Si)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="energy", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def slit_demo(E=8.0, alpha=0.3, slit_um=100.0, p_target=0.01, L_m=0.698,
              n_frames=20, frame_s=1.0, seed=1, path="slit_demo.png"):
    """Show the effect of slit height: (left) the spectrum at one energy/angle for
    this slit, (right) beam FWHM at the detector vs slit height, marking the
    Nyquist target (2 strips) and the diffraction-limited optimum.

    p_target is the pileup the mirror is calibrated to on the DIRECT beam (sample
    out) -- it is independent of the grazing angle and just scales the overall
    brightness of the spectrum."""
    sigma = physics.beam_sigma_strips(slit_um, E, L_m)

    # calibrate the mirror with this beam width, then acquire one spectrum
    h = controller.auto_align(E=E, p_target=p_target, n_frames=n_frames, frame_s=frame_s,
                              seed=seed, beam_sigma_ch=sigma, verbose=False)
    theta = h["theta"][-1]
    det = detector.MythenDetector(energy_keV=E, seed=seed, beam_sigma_ch=sigma)
    N_inc = physics.forward(theta, E)[0] / det.peak_frac
    pattern = sample.diffraction_pattern(E, alpha, N_inc, det, peak_sigma_strip=sigma)
    det.acquire_pattern(pattern, n_frames=n_frames, frame_s=frame_s)
    summed = det.frames.sum(axis=0)

    fwhm = physics.beam_fwhm_um(slit_um, E, L_m)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: the spectrum for the current slit
    ax1.semilogy(np.arange(det.nch), np.maximum(summed, 0.5), color="#c0392b", lw=0.9)
    ax1.set_xlim(390, 639); ax1.set_ylim(bottom=0.5)
    ax1.set_xlabel("detector strip"); ax1.set_ylabel("counts")
    ax1.set_title(f"Spectrum: {E:g} keV, grazing {alpha:g} deg, slit {slit_um:.0f} um, "
                  f"pileup {p_target*100:g}%\n"
                  f"beam {fwhm:.0f} um = {fwhm/physics.STRIP_PITCH_UM:.1f} strips")
    ax1.grid(True, which="both", alpha=0.25)

    # right: beam FWHM vs slit, with Nyquist target and the diffraction optimum
    slits = np.linspace(8, 220, 120)
    curve = np.array([physics.beam_fwhm_um(s, E, L_m) for s in slits])
    ax2.plot(slits, curve, color="#1f5fa8", lw=2)
    ax2.axhline(2 * physics.STRIP_PITCH_UM, ls="--", color="#2e8b57",
                label="Nyquist target (2 strips = 100 um)")
    s_opt = physics.optimal_slit_um(E, L_m)
    ax2.plot(s_opt, physics.beam_fwhm_um(s_opt, E, L_m), "v", color="#c0392b", ms=9,
             label=f"diffraction optimum ({s_opt:.0f} um)")
    ax2.axvline(slit_um, color="#888", lw=1.5, label=f"current slit ({slit_um:.0f} um)")
    ax2.set_xlabel("slit height (um)"); ax2.set_ylabel("beam FWHM at detector (um)")
    ax2.set_title("Beam size vs slit height"); ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle("Slit-height explorer", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path, dict(beam_fwhm_um=fwhm, beam_strips=fwhm / physics.STRIP_PITCH_UM,
                      delivered_flux=N_inc, optimal_slit_um=s_opt)


if __name__ == "__main__":
    print("=== Full automated acquisition: 4-10 keV, grazing 0.2-0.7 deg ===\n")
    paths, log = run_full_sequence(n_frames=20, frame_s=1.0)
    print("\n--- acquisition log ---")
    for line in log:
        print(line)
    print(f"\nWrote {len(paths)} figures: {', '.join(paths)}")
