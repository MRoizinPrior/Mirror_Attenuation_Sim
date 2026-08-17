"""
physics.py  --  deterministic physical models for the BL 3.3.2 order-sorter sim.

This module is the "ground truth" world that the (imperfect) actuator and the
(feedback) controller will interact with in later pieces.  Nothing in here knows
about control loops, noise, or repeatability -- it is pure, deterministic
input -> output physics:

      grazing angle theta  --A(theta,E)-->        two-bounce attenuation
      incident flux F0(E)  x attenuation  -->     true rate N on a detector strip
      true rate N          --paralyzable(tau)-->  measured rate M  and  pileup p

Everything is *per detector strip* and at a single photon energy E (in keV).

Deps: numpy, scipy, xraydb   (pip install xraydb scipy numpy)
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import kv
import xraydb

# --------------------------------------------------------------------------
# Fixed parameters for our specific setup (from the commissioning report)
# --------------------------------------------------------------------------
SI_DENSITY = 2.33      # g/cm^3  -- bare silicon mirrors (the tested grating/mirrors were bare Si)
EC_BEND    = 3.0       # keV     -- ALS bend-magnet critical energy (1.9 GeV, 1.25 T)
RP_SI111   = 7000      #         -- Si(111) monochromator resolving power (E/dE)

# DECTRIS MYTHEN2 (4 mm) deadtime table: photon energy [keV] -> tau [ns].
# tau is really a function of the discriminator threshold, which is set ~0.6-0.7 x E,
# so we tabulate it against photon energy and interpolate.  Note tau is flat at
# 541 ns from 4-8 keV, then drops (detector counts faster) above 8 keV.
_TAU_E  = [4.0, 4.5, 5.4, 6.5, 8.0, 8.6, 9.2, 9.8, 10.9]
_TAU_NS = [541, 541, 541, 541, 541, 315, 235, 204, 198]


def tau_seconds(E_keV):
    """Detector deadtime tau (s) at photon energy E, interpolated from the Dectris table."""
    return np.interp(E_keV, _TAU_E, _TAU_NS) * 1e-9


# --------------------------------------------------------------------------
# Beam height at the detector as a function of slit height.
# --------------------------------------------------------------------------
# The vertical slit sets the beam height.  What lands on the detector is the
# quadrature sum of three contributions:
#   * geometric   -- the slit opening itself           (grows with slit height)
#   * diffraction -- slit diffraction lambda*L/slit     (grows as the slit SHRINKS)
#   * detector    -- the finite 50 um strip pitch       (a fixed floor)
# The geometric and diffraction terms trade off, so the beam is SMALLEST at an
# intermediate slit of about sqrt(lambda*L) -- shrinking the slit past that makes
# the spot bigger, not sharper.  L is the slit-to-detector distance.
STRIP_PITCH_UM = 50.0

def beam_fwhm_um(slit_um, E_keV, L_m=0.698, pitch_um=STRIP_PITCH_UM):
    """Beam FWHM (um) at the detector for a given slit height."""
    lam_um = (12.39842 / E_keV) * 1e-4        # X-ray wavelength in um
    L_um = L_m * 1e6
    geom = slit_um                             # geometric slit image
    diff = lam_um * L_um / slit_um             # slit-diffraction broadening
    det = pitch_um                             # detector strip-pitch floor
    return np.sqrt(geom ** 2 + diff ** 2 + det ** 2)

def beam_sigma_strips(slit_um, E_keV, L_m=0.698, pitch_um=STRIP_PITCH_UM):
    """Beam width as a Gaussian sigma in detector strips (for the peak profiles)."""
    return beam_fwhm_um(slit_um, E_keV, L_m, pitch_um) / 2.355 / pitch_um

def optimal_slit_um(E_keV, L_m=0.698):
    """Diffraction-limited slit that minimizes the beam size: slit = sqrt(lambda*L)."""
    lam_um = (12.39842 / E_keV) * 1e-4
    return np.sqrt(lam_um * L_m * 1e6)


# --------------------------------------------------------------------------
# 1) Mirror attenuation  A(theta, E) = R(theta)^2   (two grazing reflections)
# --------------------------------------------------------------------------
# xraydb gives the single-surface Fresnel reflectivity R(theta,E).  Evaluating it
# over a whole angle grid is the slow part, so we compute the curve once per energy
# and cache it; later calls just interpolate.  A ~ 1 below the critical angle
# (beam is reflected/passed), and falls ~5 orders of magnitude just above it.
_refl_cache = {}


def _reflectivity_curve(E_keV, n=1200):
    """Return cached (theta_deg, R) -- single-surface Si reflectivity vs grazing angle."""
    key = round(float(E_keV), 4)
    if key not in _refl_cache:
        theta = np.linspace(0.02, 1.2, n)                        # grazing angle [deg]
        R = np.real(xraydb.mirror_reflectivity('Si', theta * np.pi / 180.0,
                                               E_keV * 1e3, SI_DENSITY))
        _refl_cache[key] = (theta, R)
    return _refl_cache[key]


def attenuation(theta_deg, E_keV):
    """Two-bounce attenuation A = R(theta)^2 of the Coddling mirror pair (0..1)."""
    theta, R = _reflectivity_curve(E_keV)
    Ri = np.interp(theta_deg, theta, R)     # single-surface reflectivity at this angle
    return Ri ** 2                          # two reflections in series


# --------------------------------------------------------------------------
# Coated-sample reflectivity (e.g. Pt-coated grating on BL 5.3.1).  Separate
# cache from the bare-Si order-sorter mirror above: this is the SAMPLE surface.
# --------------------------------------------------------------------------
_coat_cache = {}
# material densities (g/cm^3) for common grating coatings
COATING_DENSITY = {"Si": 2.33, "Pt": 21.45, "Au": 19.30, "Ni": 8.90}


def coated_reflectivity_curve(E_keV, material="Pt", density=None, n=1200):
    """Cached (theta_deg, R): single-surface reflectivity of a coated surface vs
    grazing angle.  A high-Z coating (Pt/Au) pushes the critical angle out and
    keeps R high to much steeper angles than bare Si -- the reason a coated
    grating spans far more usable stitching decades."""
    if density is None:
        density = COATING_DENSITY.get(material, 2.33)
    key = (round(float(E_keV), 4), material, round(float(density), 4))
    if key not in _coat_cache:
        theta = np.linspace(0.02, 1.2, n)
        R = np.real(xraydb.mirror_reflectivity(material, theta * np.pi / 180.0,
                                               E_keV * 1e3, density))
        _coat_cache[key] = (theta, R)
    return _coat_cache[key]


# --------------------------------------------------------------------------
# 2) Incident flux delivered to one strip, BEFORE the mirror attenuator
# --------------------------------------------------------------------------
def _G1(y):
    """Bend-magnet vertically-integrated spectral shape:  G1(y) = y * int_y^inf K_5/3(t) dt."""
    return y * quad(lambda t: kv(5.0 / 3.0, t), y, np.inf)[0]


# Normalize the source so that at 8 keV it gives 2e9 ph/s per 0.1% bandwidth,
# which is the SPECTRA value quoted in the report.
_SRC_SCALE = 2e9 / _G1(8.0 / EC_BEND)


def _window_transmission(E_keV):
    """Beamline window stack transmission: 425 um Be + 2x50 um Kapton + 10 um Al + 5 cm air."""
    def T(material, density, thickness_um):
        mu = xraydb.material_mu(material, E_keV * 1e3, density=density)   # [1/cm]
        return np.exp(-mu * thickness_um * 1e-4)                          # thickness cm
    return (T('Be', 1.85, 425) * T('kapton', 1.42, 100)
            * T('Al', 2.70, 10) * T('air', 1.2e-3, 50000))


def incident_flux(E_keV):
    """Photons/s delivered into the beam on one strip, *before* the mirror.
       = source flux  x  Si(111) bandwidth fraction  x  window transmission."""
    src = _SRC_SCALE * _G1(E_keV / EC_BEND)   # ph/s per 0.1% bandwidth
    bw = 1000.0 / RP_SI111                      # number of 0.1%-bands inside the Si(111) passband
    return src * bw * _window_transmission(E_keV)


# --------------------------------------------------------------------------
# 3) Detector response: paralyzable pileup
# --------------------------------------------------------------------------
def measured_rate(true_rate, E_keV):
    """Paralyzable detector:  M = N * exp(-N * tau)."""
    return true_rate * np.exp(-true_rate * tau_seconds(E_keV))


def pileup_fraction(true_rate, E_keV):
    """Fraction of events lost to pileup:  p = 1 - M/N = 1 - exp(-N tau)."""
    return 1.0 - np.exp(-true_rate * tau_seconds(E_keV))


def true_rate_for_pileup(p, E_keV):
    """Invert pileup: the true rate N that yields pileup fraction p.  N = -ln(1-p)/tau."""
    return -np.log(1.0 - p) / tau_seconds(E_keV)


# --------------------------------------------------------------------------
# Convenience: the full forward chain  theta -> (N, M, p)
# --------------------------------------------------------------------------
def forward(theta_deg, E_keV):
    """Given a mirror angle, return (true_rate, measured_rate, pileup) on one strip."""
    N = incident_flux(E_keV) * attenuation(theta_deg, E_keV)   # true photons/s hitting the strip
    M = measured_rate(N, E_keV)
    p = pileup_fraction(N, E_keV)
    return N, M, p


# --------------------------------------------------------------------------
# Self-test: run `python physics.py` to sanity-check against the report numbers
# --------------------------------------------------------------------------
if __name__ == "__main__":
    E = 8.0
    print(f"Self-test at E = {E} keV")
    print(f"  tau                 = {tau_seconds(E)*1e9:.0f} ns")
    print(f"  incident flux F0    = {incident_flux(E):.2e} ph/s/strip (no mirror)")
    print(f"  true rate for 1% pileup = {true_rate_for_pileup(0.01, E):.2e} ph/s")
    for th in [0.25, 0.30, 0.39, 0.50]:
        N, M, p = forward(th, E)
        print(f"  theta={th:.2f} deg -> A={attenuation(th,E):.2e}  N={N:.2e}  M={M:.2e}  pileup={p*100:.2f}%")
