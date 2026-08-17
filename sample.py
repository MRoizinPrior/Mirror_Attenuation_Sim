"""
sample.py -- the sample stage and its grazing-incidence diffraction pattern.

Two parts:

1) SampleStage -- the mechanics we automate.  The sample faces DOWN in the
   chamber on a vertical (z) motor plus a rotation.  To take it OUT of the beam
   we rotate it flat (horizontal) and lift it ~2 mm; to put it back IN we lower
   it and rotate to the grazing angle we want.  The class just tracks that state
   and logs each move, so a run reads like a real acquisition recipe.

2) diffraction_pattern -- given the beam that the mirror delivers, what does the
   MYTHEN see when the sample (a grating) is in the beam at grazing angle alpha?
   We place the diffraction orders on the detector strips using the grating
   equation, scale their brightness by the bare-Si reflectivity at that angle,
   and share intensity between orders with a simple blaze envelope.  This is a
   geometric / phenomenological model (correct order POSITIONS and the right
   reflectivity trend), not a full RCWA efficiency calculation.

Convention (from the report): alpha is the grazing angle measured from the sample
surface; order m = 0 is the specular reflection at beta = alpha.

Deps: numpy, physics.py.
"""

import numpy as np
import physics

GRATING_LMM = 200.0     # grating groove density, lines/mm (bare-Si grating from the report)
STRIP_PITCH_M = 50e-6   # MYTHEN2 strip pitch (50 um)


# ==========================================================================
# 1) The stage
# ==========================================================================
class SampleStage:
    def __init__(self, out_lift_mm=2.0):
        self.out_lift = out_lift_mm      # how far up we lift to clear the beam
        self.z_mm = 0.0                  # vertical position (0 = in-beam height)
        self.grazing_deg = 0.0           # current rotation (grazing angle when in beam)
        self.in_beam = False
        self.log = []                    # human-readable list of the moves we made

    def move_out(self):
        """Take the sample out of the beam: rotate flat (horizontal) and lift ~2 mm."""
        self.grazing_deg = 0.0
        self.z_mm += self.out_lift
        self.in_beam = False
        self.log.append(f"sample OUT  (rotate to horizontal, lift +{self.out_lift:.1f} mm)")

    def move_in(self):
        """Return the sample to beam height (still need to rotate to a grazing angle)."""
        self.z_mm -= self.out_lift
        self.in_beam = True
        self.log.append(f"sample IN   (lower -{self.out_lift:.1f} mm to beam height)")

    def rotate_to(self, alpha_deg):
        """Rotate the (in-beam) sample to grazing angle alpha."""
        self.grazing_deg = alpha_deg
        self.log.append(f"   rotate to grazing {alpha_deg:.2f} deg -> acquire")


# ==========================================================================
# 2) The diffraction pattern on the detector
# ==========================================================================
def _sample_reflectivity(E_keV, alpha_deg, material="Si", density=None):
    """Single-surface reflectivity of the sample surface at grazing angle alpha
    (0..1).  material='Si' is the bare-Si case (BL 3.3.2 commissioning grating);
    material='Pt' models the Pt-coated grating measured on BL 5.3.1, which stays
    reflective to much steeper angles."""
    if material == "Si" and density is None:
        theta, R = physics._reflectivity_curve(E_keV)
    else:
        theta, R = physics.coated_reflectivity_curve(E_keV, material, density)
    return float(np.interp(alpha_deg, theta, R))


def grating_orders(E_keV, alpha_deg, lines_per_mm=GRATING_LMM,
                   beta_max_deg=1.4, max_order=10):
    """Propagating diffraction orders as (m, exit grazing angle beta_deg).

    Grazing-incidence grating equation (angles from the surface):
        cos(beta_m) = cos(alpha) - m * lambda / d
    Orders with |cos| > 1 are evanescent (do not propagate) and are dropped.
    m = 0 is the specular beam at beta = alpha."""
    d_m = 1e-3 / lines_per_mm                       # groove period [m]
    lam = (12.39842 / E_keV) * 1e-10                # wavelength [m]  (hc = 12.398 keV.A)
    ca = np.cos(np.radians(alpha_deg))
    orders = []
    for m in range(-max_order, max_order + 1):
        cb = ca - m * lam / d_m
        if -1.0 <= cb <= 1.0:
            beta = np.degrees(np.arccos(cb))
            if 0.0 < beta <= beta_max_deg:
                orders.append((m, beta))
    return orders


def _blaze_efficiency(m, m_blaze=1, sigma=1.3):
    """Simple blaze envelope: intensity peaks at the blazed order m_blaze."""
    return np.exp(-0.5 * ((m - m_blaze) / sigma) ** 2)


def diffraction_pattern(E_keV, alpha_deg, N_incident, det,
                        lines_per_mm=GRATING_LMM, L_m=0.504, beta_ref_deg=0.45,
                        satellite_frac=0.06, peak_sigma_strip=0.7,
                        material="Si", density=None):
    """Per-strip TRUE rate (ph/s), length det.nch, for the sample at grazing alpha.

    L_m defaults to 0.504 m: the grating centre sits ~504 mm from the detector
    (measured range 379-629 mm across the grating; the exact spot where the beam
    lands, and hence L, shifts with sample angle and height -- 504 mm is the
    ideal "hit the centre" case).

    N_incident : the beam rate the mirror delivers to the sample (ph/s).
    L_m        : sample-to-detector distance; sets the strips-per-degree scale.
    beta_ref_deg : the exit angle that maps to the detector centre strip.
    satellite_frac : strength of the inter-order ('quarter-order') satellites the
                     report saw, relative to their neighbouring main order.
    """
    strips_per_deg = np.radians(1.0) * L_m / STRIP_PITCH_M     # deg -> strips on the detector
    R = _sample_reflectivity(E_keV, alpha_deg, material, density)   # overall grating reflectance
    ch = np.arange(det.nch)
    rate = np.zeros(det.nch)

    def add_peak(beta_deg, area):
        """Drop a narrow, area-normalized peak at the strip for this exit angle."""
        strip = det.beam_center + (beta_deg - beta_ref_deg) * strips_per_deg
        prof = np.exp(-0.5 * ((ch - strip) / peak_sigma_strip) ** 2)
        s = prof.sum()
        if s > 0:
            rate[:] += area * prof / s

    # --- main integer orders, shared out by the blaze envelope ---
    orders = grating_orders(E_keV, alpha_deg, lines_per_mm)
    if orders:
        effs = np.array([_blaze_efficiency(m) for m, _ in orders])
        effs = effs / effs.sum()
        for (m, beta), e in zip(orders, effs):
            add_peak(beta, N_incident * R * e)

        # --- small inter-order satellites (report's quarter-order peaks) ---
        if satellite_frac > 0:
            d_m = 1e-3 / lines_per_mm
            lam = (12.39842 / E_keV) * 1e-10
            ca = np.cos(np.radians(alpha_deg))
            for (m, _), e in zip(orders, effs):
                for frac in (0.25, 0.5, 0.75):
                    cb = ca - (m + frac) * lam / d_m
                    if -1.0 <= cb <= 1.0:
                        beta = np.degrees(np.arccos(cb))
                        add_peak(beta, N_incident * R * e * satellite_frac)

    return rate


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from detector import MythenDetector
    E = 8.0
    det = MythenDetector(energy_keV=E, seed=0)
    stage = SampleStage()
    stage.move_out(); stage.move_in()
    print("Stage moves:")
    for line in stage.log:
        print("  " + line)
    print(f"\nOrders and reflectivity at {E} keV:")
    for a in (0.2, 0.4, 0.6):
        R = _sample_reflectivity(E, a)
        ords = grating_orders(E, a)
        betas = ", ".join(f"m={m}:{b:.3f}deg" for m, b in ords[:5])
        print(f"  grazing {a:.1f} deg  R={R*100:6.3f}%  orders[{len(ords)}]: {betas}")
    # a pattern
    N_inc = physics.true_rate_for_pileup(0.01, E)
    patt = diffraction_pattern(E, 0.4, N_inc, det)
    print(f"\npattern total rate {patt.sum():.2e} ph/s, peak strip {patt.argmax()} "
          f"({patt.max():.1f} ph/s)")
