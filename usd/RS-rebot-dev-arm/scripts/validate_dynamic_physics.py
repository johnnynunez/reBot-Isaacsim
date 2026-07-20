#!/usr/bin/env python3
"""Run a short PhysX or Newton smoke test for the reBot USD asset.

Use the Isaac Sim Python launcher and point the script at the release directory:

    ISAACSIM_PATH=/path/to/isaac-sim /path/to/isaac-sim/python.sh \
        validate_dynamic_physics.py ASSET.usda newton OUTPUT.json
    ISAACSIM_PATH=/path/to/isaac-sim /path/to/isaac-sim/python.sh \
        validate_dynamic_physics.py ASSET.usda physx OUTPUT.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit(
        "usage: validate_dynamic_physics.py ASSET.usda ENGINE OUTPUT.json"
    )

asset_path = Path(sys.argv[1]).resolve()
engine = sys.argv[2]
output_path = Path(sys.argv[3])
if engine not in {"newton", "physx"}:
    raise SystemExit("ENGINE must be newton or physx")

from isaacsim import SimulationApp  # noqa: E402

isaac_sim_path = os.environ.get("ISAACSIM_PATH") or os.environ.get("ISAAC_PATH")
if not isaac_sim_path:
    raise SystemExit("set ISAACSIM_PATH to the Isaac Sim release directory")
experience = (
    Path(isaac_sim_path) / "apps/isaacsim.exp.full.newton.kit"
    if engine == "newton"
    else Path(isaac_sim_path) / "apps/isaacsim.exp.base.python.kit"
)
app = SimulationApp(
    {"headless": True, "width": 640, "height": 480},
    experience=str(experience),
)
app_utils = None
exit_code = 0

try:
    import numpy as np
    import omni.usd
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.version import get_version

    timestep = 0.002
    device = "cuda:0"
    SimulationManager.switch_physics_engine(engine)
    stage_utils.open_stage(str(asset_path))
    while stage_utils.is_stage_loading():
        app.update()
    for _ in range(10):
        app.update()

    stage = omni.usd.get_context().get_stage()

    def physics_scenes():
        return [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetTypeName() == "PhysicsScene"
        ]

    scenes_before = physics_scenes()
    SimulationManager.setup_simulation(dt=timestep, device=device)
    scenes_after = physics_scenes()

    articulation_root = next(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if any("ArticulationRootAPI" in str(schema) for schema in prim.GetAppliedSchemas())
    )
    articulation = Articulation(articulation_root)
    app_utils.play(commit=True)
    for _ in range(10):
        app.update()

    names = list(articulation.dof_names)
    expected_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint_left",
        "joint_right",
    ]
    if names != expected_names:
        raise RuntimeError(f"unexpected DOF order: {names}")

    max_efforts = articulation.get_dof_max_efforts().numpy()[0].copy()
    max_velocities = articulation.get_dof_max_velocities().numpy()[0].copy()
    expected_max_efforts = np.asarray(
        [36.0, 36.0, 36.0, 14.0, 14.0, 14.0, 500.0, 500.0]
    )
    expected_max_velocities = np.asarray(
        [50.0, 50.0, 50.0, 40.0, 40.0, 40.0, 10.0, 10.0]
    )
    lower_limits = np.asarray([-2.8, -3.14, -3.14, -1.79, -1.57, -3.14, 0.0, 0.0])
    upper_limits = np.asarray([2.8, 0.0, 0.0, 1.69, 1.57, 3.14, 0.05, 0.0715])
    position_tolerance = 1e-3
    hold_tolerance = 0.01

    start = articulation.get_dof_positions().numpy()[0].copy()
    target = np.asarray([0.0, -0.7, -1.1, 0.0, 0.0, 0.0, 0.02, 0.02])
    observed_positions = [start.copy()]

    def step(position):
        articulation.set_dof_position_targets(
            np.asarray(position, dtype=np.float32).reshape(1, -1)
        )
        app.update()
        current = articulation.get_dof_positions().numpy()[0].copy()
        observed_positions.append(current)
        return current

    ramp_steps = 250
    for index in range(ramp_steps):
        alpha = (index + 1) / ramp_steps
        step(start * (1.0 - alpha) + target * alpha)

    hold_settle_steps = 250
    for _ in range(hold_settle_steps):
        step(target)
    hold_start = articulation.get_dof_positions().numpy()[0].copy()

    hold_measurement_steps = 250
    hold_samples = [hold_start]
    for index in range(hold_measurement_steps):
        current = step(target)
        if (index + 1) % 50 == 0:
            hold_samples.append(current)
    hold_end = articulation.get_dof_positions().numpy()[0].copy()
    hold_error = np.abs(hold_end - target)
    hold_drift = np.abs(hold_end - hold_start)

    kp, kd = articulation.get_dof_gains()
    kp_array = kp.numpy().copy()
    kd_array = kd.numpy().copy()
    joint3 = names.index("joint3")
    kp_array[0, joint3] = 0.0
    kd_array[0, joint3] = 0.0
    articulation.set_dof_gains(kp_array, kd_array)
    passive_start = articulation.get_dof_positions().numpy()[0].copy()
    passive_start_velocity = articulation.get_dof_velocities().numpy()[0].copy()
    passive_probe_steps = 10
    passive_trace = [passive_start.copy()]
    for _ in range(passive_probe_steps):
        passive_trace.append(step(target))
    passive_end = articulation.get_dof_positions().numpy()[0].copy()
    passive_motion = float(passive_end[joint3] - passive_start[joint3])

    observed = np.asarray(observed_positions)
    observed_min = observed.min(axis=0)
    observed_max = observed.max(axis=0)
    positions_within_limits = bool(
        np.all(observed_min >= lower_limits - position_tolerance)
        and np.all(observed_max <= upper_limits + position_tolerance)
    )

    repo_root = Path(__file__).resolve().parents[3]
    try:
        asset_label = str(asset_path.relative_to(repo_root))
    except ValueError:
        asset_label = str(asset_path)

    active_engine = str(SimulationManager.get_active_physics_engine()).lower()
    output = {
        "asset": asset_label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "isaac_sim_version": get_version()[0],
        "timestep_s": timestep,
        "device": device,
        "requested_engine": engine,
        "active_engine": active_engine,
        "articulation_root": articulation_root,
        "dof_names": names,
        "actuator_limit_validation": "runtime readback; saturation enforcement is not tested",
        "max_efforts": max_efforts.tolist(),
        "expected_max_efforts": expected_max_efforts.tolist(),
        "max_velocities": max_velocities.tolist(),
        "expected_max_velocities": expected_max_velocities.tolist(),
        "physics_scenes_before_setup": scenes_before,
        "physics_scenes_after_setup": scenes_after,
        "position_limit_validation": "all positions observed by this smoke remain in range",
        "lower_position_limits": lower_limits.tolist(),
        "upper_position_limits": upper_limits.tolist(),
        "observed_min_positions": observed_min.tolist(),
        "observed_max_positions": observed_max.tolist(),
        "positions_within_limits": positions_within_limits,
        "start_positions": start.tolist(),
        "target_positions": target.tolist(),
        "hold_settle_steps": hold_settle_steps,
        "hold_measurement_steps": hold_measurement_steps,
        "hold_start_positions": hold_start.tolist(),
        "hold_end_positions": hold_end.tolist(),
        "hold_abs_error": hold_error.tolist(),
        "hold_max_abs_error": float(hold_error.max()),
        "hold_drift": hold_drift.tolist(),
        "hold_max_drift": float(hold_drift.max()),
        "passive_probe_joint": "joint3",
        "passive_probe_steps": passive_probe_steps,
        "passive_probe_duration_s": passive_probe_steps * timestep,
        "passive_start_velocity": passive_start_velocity.tolist(),
        "passive_probe_positions_joint3": [
            float(position[joint3]) for position in passive_trace
        ],
        "passive_motion_joint3": passive_motion,
        "passed": bool(
            active_engine == engine
            and scenes_before == ["/PhysicsScene"]
            and scenes_after == ["/PhysicsScene"]
            and np.allclose(max_efforts, expected_max_efforts, rtol=0.0, atol=1e-4)
            and np.allclose(
                max_velocities, expected_max_velocities, rtol=0.0, atol=1e-4
            )
            and hold_error.max() < hold_tolerance
            and hold_drift.max() < hold_tolerance
            and positions_within_limits
            and abs(passive_start_velocity[joint3]) < hold_tolerance
            and passive_motion > 1e-5
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output, indent=2), flush=True)
    exit_code = 0 if output["passed"] else 1
finally:
    try:
        if app_utils is not None:
            app_utils.stop()
    except Exception:
        pass
    app.close(exit_code=exit_code)

raise SystemExit(exit_code)
