#!/usr/bin/env python3
"""Isaac Sim 机械臂 + 地面 + 夹爪受力打印测试 / Isaac Sim arm + ground + gripper force test.

功能概述：
1. 启动 Isaac Sim，打开 GUI。
2. 加载 `usd/RS-rebot-dev-arm` 机械臂 USD 资产，并添加地面、灯光、相机。
3. 在夹爪的两个指尖 rigid body (`gripper_left` / `gripper_right`) 下挂载
   `isaacsim.sensors.physics.ContactSensor`，实时读取夹爪与场景其他物体
   (主要是地面) 的接触力。
4. 设置一个使夹爪略高于地面的初始姿态。当夹爪与场景发生接触 (例如用户
   在 GUI 中拖动机械臂、或通过手动调整关节使其贴地)，接触力会立刻更新；
   没有接触时合力 ≈ 0。
5. 主循环按 `PRINT_HZ` 频率持续打印：
   - 左右指尖各自的接触合力大小 (N) 与该时刻是否处于 in_contact
   - 左右指尖接触矢量之和的总力矢量 (N) 与其模长 |F|
   - 当前夹爪开度 (mm)
   关闭 Isaac Sim 窗口或 Ctrl+C 退出。

Overview:
1. Launch Isaac Sim with GUI.
2. Load the `usd/RS-rebot-dev-arm` arm USD asset, add a ground plane, lighting,
   and a view camera pointing at the workspace.
3. Attach `isaacsim.sensors.physics.ContactSensor` to both gripper finger rigid
   bodies (`gripper_left` / `gripper_right`) so we can read the contact forces
   between the gripper and scene objects (mainly the ground).
4. Set an initial pose with the gripper hovering above the ground. Whenever the
   gripper actually touches anything (e.g. the user drags it via the GUI), the
   printed contact force will rise from ~0 to >0 in real time.
5. The main loop continuously prints the per-finger scalar contact force (N),
   the vector sum (N) and its magnitude, plus the current gripper opening (mm).

推荐运行方式:
    # 终端：使用 Isaac 官方 python.sh 启动
    ./reBotArm_Isaacsim/run_gripper_force_test.sh

    # 或者直接:
    /home/<user>/IsaacSim/_build/linux-x86_64/release/python.sh \
        reBotArm_Isaacsim/isaacsim_gripper_force_test.py

停止方式：关闭 Isaac Sim 窗口或 Ctrl+C。
To stop: close the Isaac Sim window or press Ctrl+C.
"""

from __future__ import annotations

import signal
import struct
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from isaacsim import SimulationApp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "未检测到可用的 Isaac Sim Python 环境，请使用 Isaac 官方 python.sh 运行本脚本 \n"
        "No usable Isaac Sim Python environment found; please run this script with the official Isaac python.sh \n"
    ) from exc

if not callable(SimulationApp):
    raise RuntimeError(
        "检测到了不完整的 Isaac Sim Python 运行时：`SimulationApp` 不可调用，请使用 Isaac 官方 python.sh 运行本脚本 \n"
        "Incomplete Isaac Sim Python runtime detected: `SimulationApp` is not callable, \n"
        "please run this script with the official Isaac python.sh \n"
    )

# ── 路径与默认参数 ─────────────────────────────────────────────────────────────
ASSET_RELATIVE_PATH = Path("usd/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda")
GRID_TEXTURE_RELATIVE_PATH = Path("reBotArm_Isaacsim/assets/grid_ground.png")
GRID_TEXTURE_CELLS = 10
GRID_TEXTURE_SIZE = 512
GRID_TEXTURE_SCALE = np.array([10.0, 10.0], dtype=np.float64)

ROBOT_PRIM_PATH = "/World/reBotArm"
GROUND_PLANE_PRIM_PATH = "/World/defaultGroundPlane"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
DISTANT_LIGHT_PRIM_PATH = "/World/DistantLight"
DEFAULT_CAMERA_EYE = np.array([0.595, 0.532, 0.636], dtype=np.float64)
DEFAULT_CAMERA_TARGET = np.array([0.0, 0.0, 0.35], dtype=np.float64)

# 夹爪的 collider-bearing prim 位于 USD 中的实际路径。
# USD 层级 (base.usda):
#     tn__00armrs_asmv3_hJ6D/         ← 默认 prim
#         Geometry/
#             base_link/link1/link2/link3/link4/link5/link6/gripper_end/
#                 gripper_left/        ← outer Xform (纯几何/分组，无碰撞 API)
#                     gripper_left/    ← inner Xform (持有 PhysicsCollisionAPI，
#                                       由 physics.usda 进一步叠加 PhysicsRigidBodyAPI)
#                         <mesh>
#
# 注意：`add_reference_to_stage(asset, "/World/reBotArm")` 会把 defaultPrim
# 的子层级直接合成到 `/World/reBotArm` 之下，因此 stage 上的实际 prim path 是
#   /World/reBotArm/Geometry/base_link/.../gripper_end
# 而不是
#   /World/reBotArm/tn__00armrs_asmv3_hJ6D/Geometry/...
# 为了避免硬编码的脆弱性，启动时通过遍历 gripper_end 子树定位"最深的
# 拥有 PhysicsCollisionAPI 的 prim"，再用其路径在下面挂 ContactSensor。
GRIPPER_END_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end"
)
# 兜底：在少数 USD 结构里 `gripper_end` 节点可能位于不同位置；如果上面的
# 直接路径找不到，会改用全 stage 搜索 `gripper_end` prim。
# 用于遍历时唯一识别左右夹爪的关键词 (匹配 prim 名字)。
GRIPPER_FINGER_KEYWORDS = ("gripper_left", "gripper_right")

# 关节命名 + 初始姿态 (关节坐标 = 仿真坐标，即 PhysX 关节空间的直接值)。
# 用一个使夹爪自然指向下方、距离地面约 5-15cm 的姿态；当用户在 GUI 中
# 拖动机械臂、或通过其他方式让夹爪贴地/夹物体时，接触力立刻 > 0。
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = ("joint_left", "joint_right")
# 单位：弧度 (仿真坐标系)。第一个关节 0、第二个关节 -60°、第三个关节 -120°，
# 使前臂折叠向下；最后三关节保持 0。最终构型使 link6+Z 方向大致朝下。
INITIAL_ARM_JOINT_RAD = np.deg2rad([0.0, -0.0, -0.0, 0.0, 0.0, 0.0]).astype(np.float64)
INITIAL_GRIPPER_OPENING_M = 0.000 

PRINT_HZ = 20.0
CONTACT_FORCE_DT = 1.0 / 200.0   # ContactSensor 物理步长：尽量细，捕获瞬时尖峰

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 信号处理与工具函数                                                          │
# └──────────────────────────────────────────────────────────────────────────┘
_running = True


def _sigint_handler(signum, frame) -> None:
    """Ctrl+C 转为"下帧退出"，与其它接收端脚本语义一致。"""
    del signum, frame
    global _running
    print("\n[gripper-force-test] 收到 Ctrl+C，准备退出...")
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)


def _format_vec(vec: np.ndarray, width: int = 7, precision: int = 3) -> str:
    """把长度为 3 的向量格式化成 `(+1.234, -0.500, +0.000)` 这种字符串。"""
    return "(" + ", ".join(f"{v:+{width}.{precision}f}" for v in vec) + ")"


def _write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """不依赖 Pillow，手写一个最小的 8-bit RGB PNG 编码器。"""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG input must be uint8 RGB array")

    height, width = rgb.shape[:2]
    raw_data = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw_data, level=9)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _ensure_grid_texture() -> Path:
    """生成一张轻量网格贴图，给地面视觉材质用，不依赖 Nucleus。"""
    texture_path = REPO_ROOT / GRID_TEXTURE_RELATIVE_PATH
    size = GRID_TEXTURE_SIZE
    cells = GRID_TEXTURE_CELLS
    background = np.array([245, 245, 247], dtype=np.uint8)
    minor_line = np.array([220, 220, 224], dtype=np.uint8)
    major_line = np.array([190, 190, 198], dtype=np.uint8)
    image = np.tile(background, (size, size, 1))
    cell_size = size // cells

    for index in range(cells + 1):
        pos = min(index * cell_size, size - 1)
        line_color = major_line if index == cells // 2 else minor_line
        thickness = 2 if index == cells // 2 else 1
        end = min(pos + thickness, size)
        image[pos:end, :, :] = line_color
        image[:, pos:end, :] = line_color

    _write_png_rgb(texture_path, image)
    return texture_path


def _add_local_ground_plane(world) -> None:
    """和 `isaacsim_joint_receiver.py` 一致的地面创建逻辑。"""
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from isaacsim.core.api.materials.physics_material import PhysicsMaterial
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.string import find_unique_string_name

    physics_material_path = find_unique_string_name(
        initial_name="/World/Physics_Materials/physics_material",
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    physics_material = PhysicsMaterial(
        prim_path=physics_material_path,
        static_friction=0.9,
        dynamic_friction=0.7,
        restitution=0.1,
    )
    visual_material_path = find_unique_string_name(
        initial_name="/World/Looks/grid_ground_material",
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    grid_texture = _ensure_grid_texture()
    visual_material = OmniPBR(
        prim_path=visual_material_path,
        texture_path=str(grid_texture),
        texture_scale=GRID_TEXTURE_SCALE,
        color=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    ground_plane = GroundPlane(
        prim_path=GROUND_PLANE_PRIM_PATH,
        name="default_ground_plane",
        z_position=0.0,
        physics_material=physics_material,
        visual_material=visual_material,
    )
    world.scene.add(ground_plane)


def _add_default_lighting() -> None:
    """Dome + Distant 组合的基础灯光，不依赖 Nucleus。"""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, Sdf, UsdLux

    stage = get_current_stage()
    if not stage.GetPrimAtPath(DOME_LIGHT_PRIM_PATH).IsValid():
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path(DOME_LIGHT_PRIM_PATH))
        dome.CreateIntensityAttr(450.0)
        dome.CreateColorAttr(Gf.Vec3f(0.96, 0.96, 0.98))

    if not stage.GetPrimAtPath(DISTANT_LIGHT_PRIM_PATH).IsValid():
        distant = UsdLux.DistantLight.Define(stage, Sdf.Path(DISTANT_LIGHT_PRIM_PATH))
        distant.CreateIntensityAttr(500.0)
        distant.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 45.0, 0.0))


def _set_viewport_camera(sim_app) -> None:
    """GUI 准备好后，把默认 viewport 相机指向机器人工作区。"""
    from isaacsim.core.utils.viewports import set_camera_view

    for _ in range(3):
        sim_app.update()

    set_camera_view(
        eye=DEFAULT_CAMERA_EYE,
        target=DEFAULT_CAMERA_TARGET,
        camera_prim_path="/OmniverseKit_Persp",
    )


def _resolve_gripper_rigid_body_paths(
    gripper_end_prim_path: str,
    finger_keywords: tuple[str, ...],
    robot_root_prim_path: str = ROBOT_PRIM_PATH,
) -> dict[str, str]:
    """定位每个夹爪对应的 rigid body Xform (而不是内嵌 collider mesh)。

    在 USD 里夹爪这一段是这样的分层：
        gripper_end/
            gripper_left/             ← user-level Xform，带 PhysicsRigidBodyAPI，
                                         不是 instance，可以挂 ContactSensor 子节点
                gripper_left/         ← instance nested Xform
                    gripper_left       ← Mesh 几何，带 PhysicsCollisionAPI
                                         (在 instance proxy 里，authoring 不允许)

    `ContactSensor` 必须是带 `CollisionAPI` 的 prim 的子节点，所以正确的
    挂法是：
        gripper_left/ContactSensor_gripper_left
    也就要求父级是 rigid body Xform (它自身也是 collider parent)。

    Args:
        gripper_end_prim_path: 形如 `/World/reBotArm/.../gripper_end`。
        finger_keywords: 夹爪 prim 名要找的关键字。
        robot_root_prim_path: 整个机器人 prim 根路径，用于兜底搜索。

    Returns:
        `dict[keyword] -> 该夹爪可挂 ContactSensor 的 rigid body Xform 完整路径`。
    """
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdPhysics

    stage = get_current_stage()

    def _gather_rigid(root_prim: Usd.Prim) -> list[tuple[int, str, str]]:
        """遍历 root 子树返回 [(depth, keyword, path)]，仅保留有 RigidBodyAPI 的 prim。"""
        out: list[tuple[int, str, str]] = []
        if not root_prim or not root_prim.IsValid():
            return out
        for prim in Usd.PrimRange(
            root_prim,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ):
            if not prim.IsValid():
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            prim_path_str = prim.GetPath().pathString
            prim_name = prim.GetName()
            for keyword in finger_keywords:
                if (
                    prim_name == keyword
                    or prim_path_str.endswith(f"/{keyword}")
                    or f"/{keyword}/" in prim_path_str
                ):
                    out.append((prim_path_str.count("/"), keyword, prim_path_str))
                    break
        return out

    resolved: dict[str, str] = {}
    gripper_end_prim = (
        stage.GetPrimAtPath(gripper_end_prim_path)
        if is_prim_path_valid(gripper_end_prim_path)
        else None
    )

    # ── 步骤 1：以 gripper_end 为根 ──
    pool = _gather_rigid(gripper_end_prim) if gripper_end_prim else []
    all_pool = list(pool)
    for _depth, kw, path in pool:
        resolved.setdefault(kw, path)

    # ── 步骤 2：兜底 ──
    if len(resolved) < len(finger_keywords) and is_prim_path_valid(robot_root_prim_path):
        wider_pool = _gather_rigid(stage.GetPrimAtPath(robot_root_prim_path))
        for _depth, kw, path in wider_pool:
            resolved.setdefault(kw, path)
        all_pool.extend(wider_pool)

    missing = [kw for kw in finger_keywords if kw not in resolved]
    if missing:
        print(
            f"[gripper-force-test] [diag] 搜索 {gripper_end_prim_path} 失败。"
            f"已找到 {len(all_pool)} 个 RigidBodyAPI 候选 (前 20 条):"
        )
        for _depth, kw, path in all_pool[:20]:
            print(f"  · depth={_depth}, kw={kw}, path={path}")
        raise RuntimeError(
            f"未找到夹爪 rigid body: 缺少 {missing}。请检查 USD 中是否真的为"
            f"这些 prim 添加了 PhysicsRigidBodyAPI，或手动指定 rigid body 路径。"
        )

    # ── 步骤 3："最深启发式" (本仓库里夹爪只是常规命名应该一段深度，但保留) ──
    final: dict[str, str] = {}
    for kw in finger_keywords:
        same_kw = [(d, p) for (d, k, p) in all_pool if k == kw]
        if not same_kw:
            final[kw] = resolved[kw]
        else:
            same_kw.sort(key=lambda item: item[0], reverse=True)
            final[kw] = same_kw[0][1]
    return final


# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 主程序                                                                        │
# └──────────────────────────────────────────────────────────────────────────┘


def main() -> None:
    asset_path = REPO_ROOT / ASSET_RELATIVE_PATH
    if not asset_path.exists():
        raise FileNotFoundError(
            f"Isaac Sim 资产不存在: {asset_path} / "
            f"Isaac Sim asset not found: {asset_path}"
        )

    sim_app = SimulationApp({"headless": False})

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.physics import ContactSensor

    # 纹理符号链接：USD 中某些贴图按绝对路径编写，与接收端脚本保持一致的修复方式。
    expected_tex_dir = Path.home() / "reBotArm_control_py" / "config" / "RS-rebot-dev-arm" / "Textures"
    if not expected_tex_dir.exists():
        expected_tex_dir.parent.mkdir(parents=True, exist_ok=True)
        actual_tex_dir = asset_path.parent / "Textures"
        expected_tex_dir.symlink_to(actual_tex_dir)

    world = World(stage_units_in_meters=1.0)
    _add_local_ground_plane(world)
    _add_default_lighting()
    add_reference_to_stage(str(asset_path), ROBOT_PRIM_PATH)

    if not is_prim_path_valid(ROBOT_PRIM_PATH):
        raise RuntimeError(
            f"Isaac Sim 中未找到机器人 Prim: {ROBOT_PRIM_PATH} / "
            f"robot prim not found in Isaac Sim: {ROBOT_PRIM_PATH}"
        )

    # 让 gripper 左右两指互不碰撞：把两指的 collider 放进一个 PhysicsCollisionGroup
    # 并把 group 的 filteredGroups 自引用——USD 的 PhysicsCollisionGroup 机制
    # 是"colliders 列表 ∩ filteredGroups 集合内的对象不互相接触"。同组内成员之
    # 间不发生接触。arm 6 个 link 之间不受影响。
    from pxr import Sdf, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    collision_group_path = Sdf.Path("/World/GripperFingerNoCollideGroup")
    collision_group = UsdPhysics.CollisionGroup.Define(stage, collision_group_path)
    colliders_api = collision_group.GetCollidersCollectionAPI()
    if colliders_api is None:
        colliders_api = UsdPhysics.CollectionAPI.Apply(collision_group.GetPrim(), "colliders")
    colliders_rel = colliders_api.GetIncludesRel()
    if colliders_rel is None:
        colliders_rel = colliders_api.CreateIncludesRel()
    finger_collider_paths = (
        f"{ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/link4/link5/link6"
        f"/gripper_end/gripper_left/gripper_left",
        f"{ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/link4/link5/link6"
        f"/gripper_end/gripper_right/gripper_right",
    )
    for finger_collider_path in finger_collider_paths:
        if stage.GetPrimAtPath(finger_collider_path).IsValid():
            colliders_rel.AddTarget(Sdf.Path(finger_collider_path))
    filtered_groups_rel = collision_group.GetFilteredGroupsRel()
    if filtered_groups_rel is None:
        filtered_groups_rel = collision_group.CreateFilteredGroupsRel()
    filtered_groups_rel.AddTarget(collision_group_path)
    print(
        f"[gripper-force-test] PhysicsCollisionGroup 已创建于 {collision_group_path}，"
        f"包含 colliders: {colliders_rel.GetTargets()}"
    )

    articulation = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="rebotarm_force_test")
    world.scene.add(articulation)

    world.reset()
    articulation.initialize()

    dof_names = list(articulation.dof_names)
    arm_joint_indices = np.array(
        [dof_names.index(name) for name in ARM_JOINT_NAMES if name in dof_names],
        dtype=np.int64,
    )
    gripper_joint_indices = np.array(
        [dof_names.index(name) for name in GRIPPER_JOINT_NAMES if name in dof_names],
        dtype=np.int64,
    )

    print("[gripper-force-test] 机器人 DOF 顺序:", dof_names)
    print(
        "[gripper-force-test] 臂段索引: "
        + "  ".join(f"{name}={idx}" for name, idx in zip(ARM_JOINT_NAMES, arm_joint_indices))
    )
    print(
        "[gripper-force-test] 夹爪索引: "
        + "  ".join(
            f"{name}={idx}" for name, idx in zip(GRIPPER_JOINT_NAMES, gripper_joint_indices)
        )
    )

    # 初始姿态：把夹爪自然朝下、距离地面几厘米。当夹爪与场景发生接触时，
    # 接触力立刻更新；无接触时合力保持 ≈ 0。
    initial_q = np.zeros(len(dof_names), dtype=np.float64)
    if len(arm_joint_indices) == len(ARM_JOINT_NAMES):
        initial_q[arm_joint_indices] = INITIAL_ARM_JOINT_RAD
    gripper_limits = np.asarray(articulation.dof_properties["upper"])[gripper_joint_indices]
    initial_gripper = float(np.clip(INITIAL_GRIPPER_OPENING_M, 0.0, gripper_limits.min()))
    initial_q[gripper_joint_indices] = initial_gripper
    articulation.set_joint_positions(initial_q)
    articulation.set_joint_velocities(np.zeros_like(initial_q))
    articulation.set_joint_efforts(np.zeros_like(initial_q))

    # 在 gripper_end 子树中动态定位夹爪 rigid body Xform。
    # 注意：rigid body 必须是 user-level (不在 instance 里) Xform，因为
    # ContactSensor 的子 prim 不能 author 到 instance proxy 里去。
    rigid_body_paths = _resolve_gripper_rigid_body_paths(
        GRIPPER_END_PRIM_PATH, GRIPPER_FINGER_KEYWORDS
    )
    for keyword, path in rigid_body_paths.items():
        print(f"[gripper-force-test] 定位夹爪 rigid body: {keyword} → {path}")

    # 给夹爪两个 finger 各自挂上一个接触传感器。
    contact_sensors: dict[str, ContactSensor] = {}
    for finger_name in GRIPPER_FINGER_KEYWORDS:
        body_path = rigid_body_paths[finger_name]
        # ContactSensor 的 prim path 必须是 rigid body path 的子节点，不能
        # 直接 author 到 instance proxy 内部 (USD 禁止)。
        sensor_path = f"{body_path}/ContactSensor_{finger_name}"
        sensor = ContactSensor(
            prim_path=sensor_path,
            name=f"contact_sensor_{finger_name}",
            dt=CONTACT_FORCE_DT,
            min_threshold=0.0,
            max_threshold=100000.0,
            radius=-1.0,  # -1 = 全局检测，对应该 finger 的全部碰撞几何
        )
        # ContactSensor 在 initialize() 时会自动注册到 world 的 physics scene，
        # 之后 sensor.get_current_frame() 即可取回 force/in_contact/contacts。
        sensor.initialize()
        contact_sensors[finger_name] = sensor
        print(
            f"[gripper-force-test] ContactSensor 已挂载: {finger_name} → {sensor_path} "
            f"（body={body_path}）"
        )

    print(f"[gripper-force-test] 接触传感器共 {len(contact_sensors)} 个：{', '.join(contact_sensors)}")
    print("[gripper-force-test] 在 GUI 中拖动机械臂让夹爪贴地/夹物体，观察打印值变化")
    print("[gripper-force-test] 关闭 Isaac Sim 窗口或 Ctrl+C 即可退出")
    print("─" * 78)

    _set_viewport_camera(sim_app)

    period = 1.0 / PRINT_HZ
    next_print_time = time.perf_counter()
    step = 0

    while _running and sim_app.is_running():
        # 物理推进一次 (render=True 让 GUI 持续刷新)。
        world.step(render=True)
        step += 1

        now = time.perf_counter()
        if now < next_print_time:
            time.sleep(min(0.005, next_print_time - now))
            continue
        next_print_time = now + period

        # 启用 raw contact 数据，让 frame["contacts"] 携带每个接触点的
        # 位置/法向/冲量等信息；然后 get_current_frame() 同时给出标量 force 与 contacts。
        per_finger_force: dict[str, float] = {}
        per_finger_in_contact: dict[str, bool] = {}
        per_finger_vec: dict[str, np.ndarray] = {}
        for finger_name, sensor in contact_sensors.items():
            sensor.add_raw_contact_data_to_frame()
            frame = sensor.get_current_frame()
            per_finger_force[finger_name] = float(frame.get("force", 0.0))
            per_finger_in_contact[finger_name] = bool(frame.get("in_contact", False))
            finger_vec = np.zeros(3, dtype=np.float64)
            for contact in frame.get("contacts", []) or []:
                impulse = np.asarray(contact["impulse"], dtype=np.float64)
                # impulse (N·s) / dt = 力 (N)；以此估算接触点处的瞬时力矢量。
                finger_vec += impulse / CONTACT_FORCE_DT
            per_finger_vec[finger_name] = finger_vec

        total_vector = sum(per_finger_vec.values(), start=np.zeros(3, dtype=np.float64))
        total_norm = float(np.linalg.norm(total_vector))

        # 夹爪开度 (mm) 仅作可读的状态信息。
        gripper_positions = articulation.get_joint_positions(joint_indices=gripper_joint_indices)
        opening_mm = float(gripper_positions.mean()) * 1000.0

        line_parts = [f"[step {step:06d} t={step * (1.0 / 60.0):6.2f}s]"]
        for finger_name in contact_sensors:
            flag = "★" if per_finger_in_contact.get(finger_name, False) else "·"
            vec = per_finger_vec.get(finger_name, np.zeros(3))
            line_parts.append(
                f"{finger_name}={flag}{per_finger_force.get(finger_name, 0.0):6.2f}N"
                f"(vec{_format_vec(vec)})"
            )
        line_parts.append(f"|F|={total_norm:6.2f}N")
        line_parts.append(f"opening={opening_mm:5.1f}mm")
        print("  ".join(line_parts))

    print("[gripper-force-test] 仿真已停止，正在关闭...")
    sim_app.close()
    print("[gripper-force-test] 已退出。")


if __name__ == "__main__":
    main()
