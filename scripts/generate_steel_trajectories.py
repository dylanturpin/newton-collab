"""Regenerate fixed Steel robot paths: uv run --extra examples scripts/generate_steel_trajectories.py."""

import importlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.robot import _tabletop_common
from newton.viewer import ViewerNull


def main():
    wp.init()
    newton.use_coord_layout_targets = True
    trajectories = {}
    plan_ik = _tabletop_common.plan_ik
    _tabletop_common._trajectories = lambda: {}

    def record(builder, ee, offset, poses, device, desired_rotation=None, waypoint_dt=None):
        rotation = wp.quat(1.0, 0.0, 0.0, 0.0) if desired_rotation is None else desired_rotation
        key = _tabletop_common._trajectory_key(builder, ee, offset, poses, rotation, waypoint_dt)
        result = plan_ik(builder, ee, offset, poses, device, desired_rotation, waypoint_dt)
        coordinates = np.tile(builder.joint_q, (len(result), 1)).astype(np.float32)
        for index, (_, values) in enumerate(result):
            coordinates[index, : len(values)] = values
        trajectories[key] = {"times": [time for time, _ in result], "coordinates": coordinates.tolist()}
        return result

    for name in ("bolt_sweep", "wrecking_ball", "plate_rack", "pelican_bicycle", "cup_unpacking", "toy_well"):
        module = importlib.import_module(f"newton.examples.robot.example_steel_{name}")
        for value in vars(module).values():
            if isinstance(value, ModuleType) and hasattr(value, "plan_ik"):
                value.plan_ik = record
        args = module.Example.create_parser().parse_args(["--viewer", "null"])
        module.Example(ViewerNull(), args)

    destination = Path(newton.examples.get_asset("steel_tabletop/trajectories.json"))
    destination.write_text(json.dumps(trajectories, indent=2) + "\n")


if __name__ == "__main__":
    main()
