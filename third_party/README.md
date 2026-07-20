# third_party provenance

## reBotArm_control_py

- **Upstream**: <https://github.com/Seeed-Projects/reBotArm_control_py>
- **Vendoring method**: plain file snapshot (not a git submodule / subtree); the
  exact upstream base commit was not recorded at vendoring time.

### Local divergence from upstream

This snapshot is **not** a pristine copy. Known local modifications:

1. **Live `get_positions` for RobStride joints**
   (`reBotArm_control_py/actuator/rebotarm.py`, `get_positions`): reads the
   live `mechPos` parameter (`0x7019`) per RobStride joint after a
   `request_feedback` poll, instead of the cached `get_state()` value, which
   freezes at its first (or zero) value on RobStride firmware. This is the fix
   from reBot-Isaacsim PR #5.
2. **RobStride velocity caveat**
   (`reBotArm_control_py/actuator/rebotarm.py`, `get_velocities`): documents
   that the `mechVel` parameter (`0x701A`) is not rad/s on RS firmware
   (measured 2026-07-17); live velocity should be finite-differenced from
   `get_positions()`.
3. **Gravity calibration notes**: `docs/gravity_calibration_rs_2026-07-17.md`.

### Snapshot quirks

The snapshot's own `README.md` describes Isaac Sim live-mirror example scripts
(`example/11a_gravity_joint_sender.py`, `example/11b_isaacsim_joint_receiver.py`,
...) that are not present in `example/`; the maintained versions of those
scripts live in this repo at `reBotArm_Isaacsim/`.

Before re-syncing this directory from upstream, verify these modifications are
either already merged upstream or re-applied on top of the new snapshot.
