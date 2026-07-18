# Cross-engine physics parity — RS reBot DevArm

The same arm, built from the one URDF (`urdf/00-arm-rs_asm-v3`), behaves
identically across four physics environments. The invariant checked is the
**generalized gravity torque g(q)** — the holding torque each joint must
supply against gravity at rest. If g(q) matches, the mass/inertia/kinematic
model is the same everywhere; gravity, static equilibrium and drive droop
follow.

## Result

| Environment | reads | gravity torque vs reference |
|---|---|---|
| **Newton** (newton-physics/newton, standalone) | URDF | **2.5e-6 N·m** |
| **MuJoCo** (google-deepmind/mujoco) | this MJCF | **5.9e-6 N·m** |
| **Isaac Sim — PhysX** | USD asset | drive droop = g(q)/K to 3 digits |
| **Isaac Sim — Newton** | USD asset | drive droop = g(q)/K to 3 digits |
| Pinocchio (reference) | URDF | — |
| Real arm (RobStride) | hardware | PD-sweep g(q) to ~5–11% |

Max deviation of any engine from the reference, over 5 poses (home, elbow-up
L, reach, wrist-loaded, twisted): **< 6e-6 N·m**. Pinocchio is the reference
because it independently agrees with the Isaac Sim drive droop (PhysX and
Newton) and with the on-hardware PD-sweep gravity measurements (see
`third_party/reBotArm_control_py/docs/gravity_calibration_rs_2026-07-17.md`).

## Reproduce

```bash
# Newton standalone + MuJoCo + Pinocchio (one process)
python mjcf/rebot_devarm/parity_all_engines.py        # -> parity_all_engines.json

# Isaac Sim PhysX and Newton drive droop (USD asset)
ISAACSIM_PATH=<release> python.sh \
  usd/RS-rebot-dev-arm/scripts/sim_hold_test.py \
  usd/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda newton  out.json
```

## Why it took care to get right

A first MuJoCo build disagreed with the others by ~1 N·m at joint6. Cause:
MuJoCo's URDF importer mis-rotates the `gripper_end` fixed-joint inertial when
it merges it into `link6` (it drops the joint rpy `3.1416 -1.5708 0`), placing
the merged CoM at `[-0.098, -0.144, 0.002]` instead of `[0, 0, 0.047]`. The
MJCF here keeps `gripper_end` a separate welded body so every inertial comes
straight from the URDF — see README.md. That is exactly the kind of
per-engine modeling drift this parity check exists to catch.
