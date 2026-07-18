"""Build the menagerie-style rebot_devarm.xml.

Body tree, joints, inertials and coacd convex COLLISION come from
`urdf-to-mjcf` (discoverse-dev/urdf-to-mjcf, -ct convex_hull) on the URDF —
every link keeps its exact URDF inertial and gripper_end stays a separate
welded body (so the gripper_end fixed-joint inertial-merge bug MuJoCo's own
importer hits is avoided).

On top of that this script authors:
  - fixed base + base_link inertial
  - <option>, joint armature/damping defaults, position actuators (PD gains),
    home/raised keyframes
  - per-part COLOURED visual geoms parsed straight from the URDF <visual>
    list (the URDF stores every visual as flat grey; the real arm's lime
    covers / black motors / aluminium are recovered from the mesh filenames:
    *_green + gripper fingers -> lime, motor_* / *_black -> black, else grey).

Gravity parity: MuJoCo qfrc_bias vs Pinocchio g(q) < 1e-5 N.m.
Run: python build_mjcf.py
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SRC = "/tmp/u2m/rebot.xml"
URDF = HERE.parents[1] / "urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"


def colour_of(mesh_file: str) -> str:
    n = mesh_file.lower()
    if "_green" in n or n.startswith("gripper_left") or n.startswith("gripper_right"):
        return "lime"
    if "motor" in n or "_black" in n:
        return "black"
    return "alum"


def rpy_to_quat(r, p, y):
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    return (f"{cr*cp*cy + sr*sp*sy:.8g} {sr*cp*cy - cr*sp*sy:.8g} "
            f"{cr*sp*cy + sr*cp*sy:.8g} {cr*cp*sy - sr*sp*cy:.8g}")


# URDF visuals per link: (mesh_name, pos, quat, colour)
urdf = ET.parse(URDF).getroot()
link_visuals = {}
mesh_names = set()
for link in urdf.findall("link"):
    vs = []
    for v in link.findall("visual"):
        g = v.find("geometry/mesh")
        if g is None:
            continue
        mf = g.get("filename").split("/")[-1]
        o = v.find("origin")
        xyz = o.get("xyz", "0 0 0") if o is not None else "0 0 0"
        rpy = [float(x) for x in ((o.get("rpy", "0 0 0").split()) if o is not None else ["0", "0", "0"])]
        name = mf.rsplit(".", 1)[0]
        mesh_names.add((name, mf))
        vs.append((name, xyz, rpy_to_quat(*rpy), colour_of(mf)))
    link_visuals[link.get("name")] = vs

tree = ET.parse(SRC)
root = tree.getroot()
root.set("model", "rebot_devarm")

comp = root.find("compiler")
comp.set("meshdir", "assets")
comp.set("autolimits", "true")
root.insert(list(root).index(comp) + 1,
            ET.Element("option", {"integrator": "implicitfast", "cone": "elliptic", "impratio": "10"}))

wb = root.find("worldbody")
base = wb.find("body[@name='base_link']")
fj = base.find("freejoint")
if fj is not None:
    base.remove(fj)
base.insert(0, ET.Element("inertial", {"pos": "2.87642e-06 -0.000122302 0.0243376",
            "mass": "1.1774", "fullinertia": "0.0013304 0.00213119 0.00275877 1e-08 0 0"}))
for g in wb.findall("geom"):
    wb.remove(g)
for l in wb.findall("light"):
    wb.remove(l)

# replace merged single-colour visuals with per-part coloured URDF visuals
for body in root.iter("body"):
    name = body.get("name")
    for g in list(body.findall("geom")):
        cls = g.get("class")
        mesh = g.get("mesh") or ""
        if cls == "visual" or "merged" in mesh or g.get("material") == "default_material":
            body.remove(g)
    for meshname, pos, quat, col in link_visuals.get(name, []):
        body.append(ET.Element("geom", {"type": "mesh", "class": "visual",
                    "material": col, "mesh": meshname, "pos": pos, "quat": quat}))

# assets: flatten collision paths, add per-part visual meshes, colour materials
asset = root.find("asset")
for m in asset.iter("mesh"):
    m.set("file", m.get("file").split("/")[-1])
existing = {m.get("name") for m in asset.iter("mesh")}
for name, mf in sorted(mesh_names):
    if name not in existing:
        asset.append(ET.Element("mesh", {"name": name, "file": mf}))
for m in list(asset.findall("material")):
    asset.remove(m)
for nm, rgba in {"black": "0.05 0.05 0.06 1", "lime": "0.72 0.85 0.20 1",
                 "alum": "0.78 0.78 0.80 1"}.items():
    asset.insert(0, ET.Element("material", {"name": nm, "rgba": rgba,
                 "specular": "0.4", "shininess": "0.3"}))
for el in list(asset):
    if (el.tag == "texture" and el.get("type") == "skybox") or el.get("name") == "groundplane":
        asset.remove(el)

# defaults: joint dynamics + actuator classes
robot_def = root.find("default/default[@class='robot']")
ET.SubElement(robot_def, "joint", {"armature": "0.01", "frictionloss": "0.2"})
for cls, (fr, kp, kv, dmp) in {"rs06": (36, 900, 60, 5), "rs00": (14, 120, 10, 2),
                               "gripper": (500, 100, 4, 1)}.items():
    dc = ET.SubElement(robot_def, "default", {"class": cls})
    ET.SubElement(dc, "joint", {"damping": str(dmp)})
    ET.SubElement(dc, "position", {"kp": str(kp), "kv": str(kv), "forcerange": f"-{fr} {fr}"})

CLASS = {"joint1": "rs06", "joint2": "rs06", "joint3": "rs06", "joint4": "rs00",
         "joint5": "rs00", "joint6": "rs00", "joint_left": "gripper", "joint_right": "gripper"}
for joint in root.iter("joint"):
    if joint.get("name") in CLASS:
        joint.set("class", CLASS[joint.get("name")])
act = root.find("actuator")
for m in list(act):
    act.remove(m)
for n, cls in CLASS.items():
    ET.SubElement(act, "position", {"class": cls, "name": n, "joint": n})

kf = ET.SubElement(root, "keyframe")
ET.SubElement(kf, "key", {"name": "home", "qpos": "0 " * 7 + "0", "ctrl": "0 " * 7 + "0"})
ET.SubElement(kf, "key", {"name": "raised", "qpos": "0 -0.7 -1.1 0 0 0 0 0",
              "ctrl": "0 -0.7 -1.1 0 0 0 0 0"})

for v in root.findall("visual")[1:]:
    root.remove(v)

ET.indent(tree, space="  ")
tree.write(HERE / "rebot_devarm.xml", encoding="unicode", xml_declaration=False)
print("WROTE rebot_devarm.xml,", sum(len(v) for v in link_visuals.values()), "coloured visual geoms")
