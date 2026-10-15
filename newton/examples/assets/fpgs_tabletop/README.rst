FPGS tabletop assets
====================

The three examples run on CUDA through SolverFeatherPGS's matrix-free path::

    uv run -m newton.examples fpgs_bolt_sweep
    uv run -m newton.examples fpgs_wrecking_ball
    uv run -m newton.examples fpgs_plate_rack

Use the side panel to stop automatic robot commands, move the robot path
slider, or reset. Drag dynamic bodies with the viewer's picking tool to apply
forces. The path slider changes actuator targets; it never overwrites object
poses. The crane's slew, boom, chain and ball are passive. Its arm releases
the yellow side handle after slewing. Its broad opposing faces let the
gripper turn the crane through contact without a grasp weld or crane motor.
A 3 N·m dry-friction slew brake resists back-driving by the swinging ball;
it yields when the gripper supplies enough torque. All three rack plates are free bodies.

The default budget is four substeps at 60 Hz and 12 PGS iterations; the
wrecking ball uses 32 iterations so the struck ball does not gain spin. Override
with ``--substeps``, ``--iterations`` and ``--contact-response`` (the bolt sweep
defaults to ``propagation_colored``). Run ``--viewer null --test`` for the
complete 900-frame outcome checks. These check collection, lift/placement,
collapse/chain attachment, finite state and solver contact-row capacity.
CUDA graphs reuse the simulation work; CPU matrix-free execution is not
supported by FPGS. Asset downloads and IK initialization occur before stepping.

Each asset is a binary USD file using standard UsdPhysics schemas: rigid
bodies with authored mass, center of mass and principal inertia; invisible
convex-hull collision meshes; visual meshes with UsdPreviewSurface materials;
and, for the crane, fixed and revolute joints plus collision filter pairs.
Sizes are baked at the scale each scene uses. The examples load each file once
with ``ModelBuilder.add_usd`` and copy it per instance, so repeated fasteners
share their meshes.

Provenance
----------

These meshes were generated for the original montage with LuckyIYI/proc-gen-3d
at ``45739ebfef8a8e933ba08b0c087618cd14dc54c9`` and archived on
``codex/physics-scene-archive``, then converted to USD: crane (hard toy
studio), plate/rack (plate studio), and fasteners (hardware).
The convex decomposition retains nut bores and rack openings. Thread-scale
contact is approximated by the convex pieces; visual threads remain detailed.

Robot assets are downloaded separately, pinned to Newton assets
``f8fb7abcbeba2318814a74f3eeb02780ad7925d6`` (FR3 and hand) and MuJoCo Menagerie
``f054586a8e90465d49ee5be15335c4a0c7f57caf`` (UR10e). Their upstream licenses
remain with the downloaded packages. No robot mesh is copied into this folder.
