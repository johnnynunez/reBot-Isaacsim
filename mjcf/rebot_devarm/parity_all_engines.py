"""Cross-engine gravity parity: Newton (standalone) vs MuJoCo vs Pinocchio,
with the Isaac Sim PhysX / Newton results cross-referenced from the USD
drive-droop validation.

Metric: the generalized gravity torque g(q) at rest (the physics invariant
the arm must reproduce identically everywhere). For each engine we read the
SAME URDF-derived model and compute the holding torque at several poses.

Newton standalone: eval_inverse_dynamics_passive gravity_force (reads URDF).
MuJoCo:            qfrc_bias at qvel=0 (reads the menagerie MJCF).
Pinocchio:         computeGeneralizedGravity (reads URDF) -- reference; it
                   matches Isaac Sim PhysX+Newton drive droop and the real-arm
                   PD-sweep measurements to 3 digits.

Run: python parity_all_engines.py   (writes parity_all_engines.json)
"""
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import mujoco
import warp as wp
import newton

HERE = Path(__file__).resolve().parent
URDF = str(HERE.parents[1] / "urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf")
MJCF = str(HERE / "rebot_devarm.xml")

POSES = {
    "home":  [0, 0, 0, 0, 0, 0],
    "L":     [0, -0.7, -1.1, 0, 0, 0],
    "reach": [0.3, -1.2, -0.8, 0.4, 0.5, 0],
    "wrist": [0, -0.9, -1.4, 0.6, 0.9, 0],
    "twist": [1.0, -0.5, -2.0, -0.8, -1.2, 1.5],
}

# --- Pinocchio (reference) ---
pm = pin.buildModelFromUrdf(URDF)
pd = pm.createData()
def g_pin(q6):
    q = np.zeros(pm.nq); q[:6] = q6
    return pin.computeGeneralizedGravity(pm, pd, q)[:6]

# --- MuJoCo ---
mm = mujoco.MjModel.from_xml_path(MJCF)
md = mujoco.MjData(mm)
def g_mjc(q6):
    md.qpos[:6] = q6; md.qpos[6:] = 0; md.qvel[:] = 0
    mujoco.mj_forward(mm, md)
    return md.qfrc_bias[:6].copy()

# --- Newton standalone ---
nb = newton.ModelBuilder(); nb.add_urdf(URDF, floating=False)
nmodel = nb.finalize()
ndof = nmodel.joint_dof_count
def g_newton(q6):
    s = nmodel.state()
    q = s.joint_q.numpy().copy(); q[:6] = q6; q[6:] = 0
    s.joint_q.assign(wp.array(q, dtype=wp.float32)); s.joint_qd.zero_()
    newton.eval_fk(nmodel, s.joint_q, s.joint_qd, s)
    gf = wp.zeros(ndof, dtype=wp.float32)
    newton.eval_inverse_dynamics_passive(nmodel, s, gravity_force=gf)
    return gf.numpy()[:6].copy()

out = {"metric": "generalized gravity torque g(q) at rest [N.m]",
       "reference": "pinocchio (== Isaac PhysX+Newton droop == real arm PD-sweep, 3 digits)",
       "engines": {"newton_standalone": newton.__version__, "mujoco": mujoco.__version__,
                   "pinocchio": pin.__version__},
       "poses": {}}
mx_n = mx_m = 0.0
for name, q6 in POSES.items():
    gp, gm, gn = g_pin(q6), g_mjc(q6), g_newton(q6)
    dn = float(np.abs(gn - gp).max()); dm = float(np.abs(gm - gp).max())
    mx_n = max(mx_n, dn); mx_m = max(mx_m, dm)
    out["poses"][name] = {
        "pinocchio": [float(round(float(x), 4)) for x in gp],
        "mujoco": [float(round(float(x), 4)) for x in gm],
        "newton_standalone": [float(round(float(x), 4)) for x in gn],
        "newton_vs_ref_max": round(dn, 8), "mujoco_vs_ref_max": round(dm, 8),
    }
    print(f"{name:6s}  newton-ref {dn:.2e}  mujoco-ref {dm:.2e}")
out["newton_standalone_max_diff"] = round(mx_n, 8)
out["mujoco_max_diff"] = round(mx_m, 8)
print(f"\nNewton standalone vs reference: max {mx_n:.2e} N.m")
print(f"MuJoCo            vs reference: max {mx_m:.2e} N.m")
print("Isaac PhysX / Isaac Newton: drive droop matches g(q)/K to 3 digits (evidence/sim_hold_*.json)")
json.dump(out, open(HERE / "parity_all_engines.json", "w"), indent=1)
print("WROTE parity_all_engines.json")
