#!/usr/bin/env python3
"""诊断脚本：在不挂 ContactSensor 的前提下，列出 /World/reBotArm 子树中所有
带 PhysicsCollisionAPI / PhysicsRigidBodyAPI 的 prim，让我们可以肉眼看出
"夹爪 collider 真正在哪"。

Diagnose without ContactSensor: list every prim under /World/reBotArm with
PhysicsCollisionAPI or PhysicsRigidBodyAPI, so we can see where the gripper
colliders really live.

用法 / Usage:
    /home/<user>/IsaacSim/_build/linux-x86_64/release/python.sh \
        reBotArm_Isaacsim/isaacsim_probe_gripper_prims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_RELATIVE_PATH = Path("usd/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda")
ROBOT_PRIM_PATH = "/World/probeRobot"

try:
    from isaacsim import SimulationApp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "未检测到 Isaac Sim Python 环境，请使用 Isaac 官方 python.sh 运行本脚本 \n"
        "No usable Isaac Sim Python environment found; please run this script with the official Isaac python.sh \n"
    ) from exc


def main() -> None:
    asset_path = REPO_ROOT / ASSET_RELATIVE_PATH
    sim_app = SimulationApp({"headless": True})

    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import Usd, UsdPhysics

    add_reference_to_stage(str(asset_path), ROBOT_PRIM_PATH)

    if not is_prim_path_valid(ROBOT_PRIM_PATH):
        print(f"[probe] 未找到 prim {ROBOT_PRIM_PATH}")
        sim_app.close()
        return

    stage = get_current_stage()
    root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)

    # 把结果写到文件 + 终端，避免 kit 在 headless 模式下吞掉 stdout。
    out_path = Path("/tmp/probe_gripper_prims.out")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"[probe] 加载完成，开始遍历 {ROBOT_PRIM_PATH} 子树 ...\n\n")
        f.write("=== PhysicsCollisionAPI 节点 ===\n")
        n_collision = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                n_collision += 1
                f.write(f"  C {prim.GetPath()}  type={prim.GetTypeName()} name={prim.GetName()}\n")
        f.write(f"共 {n_collision} 个 CollisionAPI 节点\n\n")

        f.write("=== PhysicsRigidBodyAPI 节点 ===\n")
        n_rigid = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                n_rigid += 1
                f.write(f"  R {prim.GetPath()}  type={prim.GetTypeName()} name={prim.GetName()}\n")
        f.write(f"共 {n_rigid} 个 RigidBodyAPI 节点\n\n")

        f.write("=== gripper 相关节点 (任意节点路径里出现 gripper_* 且带任意一种 PhysX API) ===\n")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if not prim.IsValid():
                continue
            prim_path_str = prim.GetPath().pathString
            if "gripper" not in prim_path_str:
                continue
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if not (has_collision or has_rigid):
                continue
            marker = ("C" if has_collision else "_") + ("R" if has_rigid else "_")
            f.write(f"  {marker} {prim_path_str}  type={prim.GetTypeName()} name={prim.GetName()}\n")
    print(f"[probe] 输出已写入 {out_path}")
    # 把关键内容再回放到 stdout，kit 在 headless 下经常吞 stdout，所以同时写盘
    try:
        sys.stdout.write(open(out_path).read())
        sys.stdout.flush()
    except Exception:
        pass

    sim_app.close()


if __name__ == "__main__":
    main()
