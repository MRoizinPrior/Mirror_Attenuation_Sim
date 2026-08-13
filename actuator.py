"""
actuator.py -- Thorlabs PIM05 piezo-inertia ("slip-stick") mirror mount.

This is the real hardware we plan to use, and the reason we need a feedback loop.
The PIM05 is a Ø1/2" mirror mount whose two adjusters are piezo-inertia motors:
a sawtooth drive makes the jaws "stick" then "slip" on an 80 TPI lead screw,
tipping the whole mount.  It steps the MIRROR ANGLE directly (no external lever
arm), so we model angle in microradians throughout.

Geometry note (grazing angle maps 1:1 to a step): the order-sorter is a pair of
parallel silica mirrors on one rigid mount, so the beam double-bounces.  Each
bounce deflects the beam by 2*theta, but the two deflections cancel -> the output
beam stays parallel to the input (only a tiny angle-dependent lateral offset).
Rotating the rigid pair by d(theta) changes the grazing angle on BOTH surfaces by
exactly d(theta), so one 0.5 urad step = 0.5 urad of grazing angle.  The familiar
"reflected ray moves 2x the mirror rotation" is a beam-POINTING effect and does
not enter the grazing angle, so there is no factor of 2 here.

Datasheet specs folded in (Thorlabs PIM05):
  * typical step size          0.5 urad
  * step size varies by +/-20%, open-loop, NOT repeatable   (datasheet note c)
  * typical angular velocity   0.05 rad/min  -> ~1667 steps/s continuous
  * total angular range        +/-2 deg (piezo + manual)
  * self-locking at rest, holds position with no power

The datasheet itself says the fix for the non-repeatability is external feedback
("To help overcome this variance, an external feedback system will need to be
used.") -- which is exactly the controller in Piece 4.

Not on the datasheet (so flagged as estimates): the slip-stick reversal lost
motion (a small directional backlash) and a slow residual thermal tip of the
mount.  Because the actuator self-locks, we keep the drift small; the larger beam
drift seen in commissioning was downstream detector-mount expansion, which moves
the beam spot but not the mirror angle.

The controller may command steps and read the detector -- it may NEVER read
theta_deg (there is no angle encoder on a PIM05).  We keep the true angle
internally only so the physics can use it and we can plot ground truth.
"""

import numpy as np


class PIM05Actuator:
    def __init__(self,
                 step_urad=0.5,               # PIM05 typical angular step
                 step_variance=0.20,          # datasheet: step size may vary by 20%, not repeatable
                 backlash_urad=1.0,           # slip-stick reversal lost motion (ESTIMATE, not on datasheet)
                 drift_urad_per_s=0.05,       # residual thermal tip of the self-locking mount (ESTIMATE)
                 angular_velocity_rad_min=0.05,  # PIM05 typical slew -> sets the step rate
                 range_deg=2.0,               # +/-2 deg total mount range
                 theta0_deg=0.35,             # starting mirror grazing angle
                 seed=None):
        self.step_urad = step_urad
        self.step_var  = step_variance
        self.b_urad    = backlash_urad          # backlash deadband width
        self.drift_rate = drift_urad_per_s      # urad/s
        self.range_urad = np.radians(range_deg) * 1e6
        self.rng = np.random.default_rng(seed)

        # Continuous step rate from the datasheet slew speed:
        #   steps/s = (angular velocity) / (step size)
        self.step_rate_hz = (angular_velocity_rad_min / 60.0) / (step_urad * 1e-6)

        # State, all in microradians of MIRROR ANGLE:
        #   _M : mirror (output) angle  -- what the physics actually sees
        #   _D : drive angle            -- leads/lags _M by up to the backlash
        #   _drift : accumulated thermal tip, added on top of _M
        x0          = np.radians(theta0_deg) * 1e6
        self._M     = x0
        self._D     = x0
        self._drift = 0.0
        self._last_dir = 0
        self.t = 0.0
        self.total_steps = 0                    # lifetime commanded steps (travel/wear/time)

    # ------------------------------------------------------------------
    # The ONLY control input: command a signed number of steps.
    # ------------------------------------------------------------------
    def step(self, n_steps):
        """Command n_steps (signed int).  Per-step +/-20% jitter + reversal backlash."""
        n_steps = int(n_steps)
        if n_steps == 0:
            return
        direction = 1 if n_steps > 0 else -1
        self._last_dir = direction

        for _ in range(abs(n_steps)):
            # each physical step is 0.5 urad perturbed by up to +/-20%, drawn fresh
            # every step -> the motion is not repeatable (datasheet note c)
            actual = self.step_urad * (1.0 + self.rng.uniform(-self.step_var, self.step_var))
            self._D += direction * actual

            # Backlash as a symmetric deadband of width b: the mirror only moves once
            # the drive pushes past the +/- b/2 slack.  On a reversal the drive must
            # first cross the whole slack zone before the mirror moves again.
            if self._D - self.b_urad / 2 > self._M:
                self._M = self._D - self.b_urad / 2
            elif self._D + self.b_urad / 2 < self._M:
                self._M = self._D + self.b_urad / 2
            # else: within the slack zone -> mirror does not move (lost motion)

            # respect the +/-2 deg mechanical range
            self._M = float(np.clip(self._M, -self.range_urad, self.range_urad))

        self.total_steps += abs(n_steps)

    def advance_time(self, dt_s):
        """Let the clock run (e.g. during an exposure); residual thermal tip accrues."""
        self._drift += self.drift_rate * dt_s
        self.t += dt_s

    # ------------------------------------------------------------------
    # GROUND TRUTH -- physics uses this; the controller in Piece 4 must NOT.
    # ------------------------------------------------------------------
    @property
    def theta_deg(self):
        """Actual mirror grazing angle (deg), including thermal tip. Hidden from the loop."""
        return np.degrees((self._M + self._drift) * 1e-6)   # urad -> rad -> deg

    def deg_per_step(self):
        """Nominal angle change per step (deg) -- an *estimate* only (real steps vary)."""
        return np.degrees(self.step_urad * 1e-6)


# Backwards-compatible alias (earlier pieces imported PicomotorActuator)
PicomotorActuator = PIM05Actuator


# --------------------------------------------------------------------------
# Self-test: demonstrate the datasheet behaviour.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    a = PIM05Actuator(seed=0)
    print("PIM05 model:")
    print(f"  step size      {a.step_urad:.2f} urad/step  (datasheet 0.5 urad)")
    print(f"  step rate      {a.step_rate_hz:.0f} steps/s  (from 0.05 rad/min slew)")
    print(f"  range          +/-{np.degrees(a.range_urad*1e-6):.1f} deg\n")

    # 1) NON-REPEATABILITY: same command from the same start, different result.
    print("Non-repeatability -- command +2000 steps five times from the same start:")
    for k in range(5):
        act = PIM05Actuator(theta0_deg=0.35, seed=None)
        act.step(2000)
        print(f"  run {k}: theta = {act.theta_deg:.5f} deg")

    # 2) BACKLASH: +1000 then -1000 does not return to start.
    print("\nBacklash on reversal (+1000 then -1000 steps):")
    act = PIM05Actuator(theta0_deg=0.35, seed=1)
    th0 = act.theta_deg
    act.step(1000);  th1 = act.theta_deg
    act.step(-1000); th2 = act.theta_deg
    net_urad = (th2 - th0) * 1e6 * np.pi / 180
    print(f"  start {th0:.5f} -> +1000 {th1:.5f} -> -1000 {th2:.5f} deg (net {net_urad:+.1f} urad)")

    # 3) DRIFT: no commands, just time.
    print("\nResidual thermal tip with no commands (100 s):")
    act = PIM05Actuator(theta0_deg=0.35, seed=2)
    th0 = act.theta_deg
    act.advance_time(100.0)
    print(f"  theta {th0:.5f} -> {act.theta_deg:.5f} deg over {act.t:.0f} s")
