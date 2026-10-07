"""Inspect cached robot assets and CUDA/OpenGL support for the hero prototype."""

import json
import os
from pathlib import Path

import warp as wp

import newton
import newton.utils
from newton.viewer import ViewerGL


def main():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyglet

    pyglet.options["headless"] = True

    wp.init()
    wp.set_device("cuda:0")
    assets = {}
    for name in ["franka_emika_panda", "unitree_g1", "unitree_go2", "universal_robots_ur5e"]:
        assets[name] = str(newton.utils.download_asset(name))
    for name in ["kuka_iiwa_14", "wonik_allegro", "ur5e_menagerie"]:
        assets[name] = str(
            Path("assets/menagerie", "universal_robots_ur5e" if name == "ur5e_menagerie" else name).resolve()
        )
    Path("assets.json").write_text(json.dumps(assets, indent=2))
    for name, path in assets.items():
        print(name, path, [str(p.relative_to(path)) for p in Path(path).rglob("*.xml")][:12], flush=True)
        print("Policies", list(Path(path).rglob("*.onnx")), flush=True)

    for name, file, fmt in [
        ("franka_emika_panda", "urdf/fr3_franka_hand.urdf", "urdf"),
        ("kuka_iiwa_14", "iiwa14.xml", "mjcf"),
        ("wonik_allegro", "right_hand.xml", "mjcf"),
    ]:
        b = newton.ModelBuilder()
        getattr(b, "add_" + fmt)(str(Path(assets[name]) / file), floating=False)
        print(
            name,
            "bodies",
            list(enumerate(b.body_label)),
            "joints",
            list(enumerate(b.joint_label)),
            "q",
            b.joint_q,
            "qstart",
            b.joint_q_start,
            flush=True,
        )

    b = newton.ModelBuilder()
    b.add_shape_box(-1, hx=0.4, hy=0.4, hz=0.4, color=(0.06, 0.62, 0.45))
    b.add_ground_plane()
    m = b.finalize()
    v = ViewerGL(width=640, height=360, headless=True)
    v.set_model(m)
    v.set_camera(wp.vec3(2, -3, 2), pitch=-25, yaw=125)
    v.begin_frame(0)
    v.log_state(m.state())
    v.end_frame()
    from PIL import Image

    Image.fromarray(v.get_frame().numpy()).save("probe.png")
    print("CUDA GL probe complete", flush=True)


if __name__ == "__main__":
    main()
