# Mirror attenuator auto-alignment simulator

A closed-loop simulation of the BL 3.3.2 portable grating diffractometer's mirror
order-sorter. It tests one question before we ask the controls group to integrate
the picomotors and the DECTRIS MYTHEN2 into LabVIEW:

> The mirror actuator has **no angle readout and is not repeatable**. Can a
> **detector-feedback loop** still park the mirror at the correct grazing angle,
> at any energy, holding a chosen detector pileup between 1% and 10%?

The answer the sim gives: **yes.** Using the detector as a flux sensor, the loop
reconstructs the deterministic energy → angle operating recipe blind, to an RMS of
well under 1 mdeg across 4–10 keV, and holds the pileup target against thermal
drift.

## Run it live in your browser (no install)

Click the badge to launch the interactive notebook on Binder — it builds a full
Python environment in the cloud and opens `Mirror_Sim_Interface.ipynb` with all the
sliders. Nothing to install; the first launch takes a minute or two.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mroizinprior/Mirror_attenuation_sim?labpath=Mirror_Sim_Interface.ipynb)

Prefer to run it locally instead? See "Quick start" below.

## The pieces

The code is deliberately split so each layer can be read and trusted on its own.

| File | Role | One-line summary |
|------|------|------------------|
| `physics.py`   | ground truth | angle → two-bounce attenuation → delivered flux → detector rate & pileup. Pure deterministic physics (xraydb reflectivity, bend-magnet spectrum, DECTRIS deadtime table). |
| `actuator.py`  | the hardware limit | **Thorlabs PIM05** piezo-inertia mirror mount: tips the mirror directly at ~0.5 µrad/step, with the datasheet ±20% non-repeatable step, reversal backlash, small residual thermal tip, ±2° range, and **no position readout**. |
| `detector.py`  | the measurement | turns a true rate into realistic MYTHEN2 output: per-strip Poisson counts + paralyzable pileup over `n_frames` frames of `frame_s` each, written in the **real `Acquisition/Frame####.dat` + `.cfg` format**, with the DECTRIS deadtime correction. |
| `controller.py`| the auto-aligner | drives the actuator using **only** detector reads (never the true angle): geometric bracket → secant trim to the target pileup, plus a hold phase that trims out drift. `budget()` reports the frames/steps/time cost to converge. |
| `run_sim.py`   | the demonstration | sweeps 4–10 keV (blind loop vs recipe), plus a convergence **budget** comparing cold vs warm start. |
| `sample.py`    | the sample | `SampleStage` (out = rotate flat + lift 2 mm; in; rotate to grazing angle) and a grazing-incidence 200 ℓ/mm grating diffraction model (bare Si *or* Pt-coated) that places orders on the detector strips. |
| `acquisition.py`| the full run | the automated recipe: per energy, sample-out mirror calibration then a sample-in grazing scan (0.2–0.7°), producing six spectra per energy, a rocking-curve summary, and (optionally) real-format `Acquisition/Frame####.dat` + `.cfg` files. |
| `stitch.py`    | dynamic-range stitching | records a synthetic pattern at several **mirror-attenuation levels** (~10× apart), drops the saturated points, and **stitches** the exposures by fitting the (unknown, imperfect) intensity ratios from the overlaps — trading a huge single-exposure S/N range for a nearly uniform one. |
| `auto_scan.py` | automated grating scan | the full **attenuation-outer / angle-inner** scan on a Pt-coated grating: the mirror sweeps attenuation **monotonically and never returns** (safe for the non-repeatable PIM05), holding still while the sample rotates through every grazing angle; each angle's on-scale levels are stitched blind and **I0-normalized to efficiency**. Knobs: pileup cap, floor S/N, bracket ratio (all sliders in the notebook). |

### Modeled hardware: Thorlabs PIM05

The actuator is modeled on the PIM05 piezo-inertia ("slip-stick") mirror mount:
0.5 µrad typical step, **±20% and not repeatable** (datasheet), 0.05 rad/min slew
(≈1667 steps/s), ±2° range, self-locking at rest. The datasheet itself prescribes
the fix we implement: *"the achieved step size is not repeatable… an external
feedback system will need to be used."* Backlash and residual thermal tip are not
on the datasheet and are flagged as estimates in the code.

Data flows one way: `physics` (truth) → `actuator` (imperfect position) →
`detector` (noisy measurement) → `controller` (blind feedback). Only `sense()` in
`controller.py` touches the true angle, and only to emulate the physical world; the
control logic itself is blind.

## Quick start

```bash
pip install -r requirements.txt

python physics.py       # self-test vs the commissioning report numbers
python actuator.py      # shows non-repeatability, backlash, drift
python detector.py      # writes a real-format acquisition + a plot
python controller.py    # aligns to 1%, 5%, 10% pileup; hold/drift demo
python run_sim.py       # energy sweep: blind loop vs the operating recipe
```

Each script writes its own `*.png`; `run_sim.py` also drops real-format
`sim_acquisitions/Acquisition9NN/` folders at each locked point.

## Key knobs

- **Pileup target** (`p_target`, 0.01–0.10) — the spec the loop servos to.
- **Frame length & count** (`frame_s`, `n_frames`) — exposure per frame and frames
  per measurement; trade statistics against speed.
- **Actuator realism** (`step_urad`, `step_variance`, `backlash_urad`,
  `drift_urad_per_s`, `angular_velocity_rad_min`) — PIM05 datasheet values;
  override with measured numbers once we characterize the mount.
- **Warm start** (`warm_start=True` in `budget_sweep`, or pass an existing
  actuator to `auto_align(act=...)`) — begin each alignment from the previous
  point's position instead of a cold 0.30°. On a fine energy scan this cuts the
  per-point motor time ~75%; on a coarse sweep it mainly helps the middle
  energies (the first point has no history to warm-start from).

## Notes and next steps

- The mirrors are modeled as **bare Si** (the tested grating turned out uncoated);
  swap the density/material in `physics.py` for a coated case.
- In the hold phase the loop trims on nearly every check because the tolerance is
  tight relative to the per-measurement noise. Widen `tol_rel` or raise `n_frames`
  to quiet the chatter.
- This is an offline de-risking model, not beamline control code. It exists to
  size the loop (measurement time, step budget, tolerance) before controls
  integration.
