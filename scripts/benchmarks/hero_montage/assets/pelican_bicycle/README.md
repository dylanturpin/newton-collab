# Pelican / Pedal Machine

A wheel-driven, stand-mounted penny-farthing automaton. Turning the large front wheel pedals the pelican's two articulated legs, drives a half-speed camshaft, flaps both wings, and opens the lower bill. The wheel is the only actuated coordinate. The fixed body, long upper bill, and rigid lower pouch make the figure recognizable as a pelican.

This package contains actual B-rep CAD, an offline interactive model, an executed custom rigid-body contact simulation, a rendered movie of that simulation, and three **untested MuJoCo scene exports**. It is a simulation/kinematic prototype, not a fabrication-ready product or a self-balancing rolling bicycle.

## Open first

- `pelican_interactive.html`: self-contained interactive actual-CAD model; open in a desktop browser. It has no network dependencies. Change the wheel-speed slider, select a view, drag to orbit, or use **Flick & coast** to remove the motor and give the wheel an initial spin.
- `media/pelican_contact_motion.mp4`: 20-second movie. The first 16 seconds replay the executed variable-speed simulation; the final four seconds replay part of it with the stationary body and wheel rims hidden so the mechanisms are visible.
- `media/mechanism_overview.png`: annotated assembly and mechanism overview.
- `cad/pelican_bicycle.step`: colored assembly of 38 actual CAD components. STEP contains geometry, not simulator joint constraints. Use `sim/joint_map.json`, `design.py`, or the MJCF files for articulations and local frames.

The interactive viewer uses WebGL where available and a Canvas2D triangle-rendering fallback otherwise. The JavaScript simulation and Canvas2D renderer were executed in headless Chromium here; the WebGL rendering path was not exercised. On slower machines the fixed-step simulation can run slower than wall time, without skipping physics steps.

## One input, three visible actions

```
Large front wheel / axle
    |
    +-- two opposite 22 mm pedal cranks
    |       +-- ankle pin -> shin -> knee -> thigh -> fixed hip
    |           (one real closed leg linkage on each side)
    |
    +-- 24-tooth gear -> 48-tooth gear / camshaft
                            |
                            +-- eccentric cam -> guided fork -> wing shoes
                            |                                  -> wings lift
                            |                          gravity -> wings return
                            |
                            +-- dwell cam -> guided pushrod -> rear jaw arm
                                                        -> lower bill opens
                                               counterweight -> bill closes
```

There is no hidden wing, jaw, or knee motor. The rear wheel has a free hinge but is unpowered and is at rest in the movie. A fixed display stand supports both axle locations: this machine demonstrates riding in place rather than rolling across a floor.

### Why a penny-farthing

Its pedals can be directly on the large wheel axle. This avoids introducing a chain or belt with unvalidated tension, tooth engagement, or extra moving bodies. Two external spur gears reduce the camshaft speed by 2:1 and reverse its rotation. Both cams are rigidly attached to that output shaft.

The entire output shaft and cam module sit outside the wheel plane, on the +Y side. A first packaging iteration had a shaft crossing the large wheel rim; the revised CAD does not. The final shaft spans Y = 18 to 84 mm, with bearings centered at Y = 22 and 78 mm. The gear faces span Y = 31 to 37 mm; the wing and beak cams are centered at Y = 50 and 65 mm.

## Dimensions and timing

CAD dimensions are in millimetres. Physics and MJCF use metres, kilograms, and seconds. X is forward, Y is lateral, and Z is upward.

| Feature | Value |
|---|---:|
| Display base | 290 x 176 x 12 mm |
| Large / small wheel diameter | 152 / 60 mm |
| Large wheel center, X/Z | 38 / 90 mm |
| Small wheel center, X/Z | -111 / 44 mm |
| Hip location, X/Z | -10 / 188 mm |
| Pedal crank radius | 22 mm |
| Thigh / shin pivot distance | 68 / 76 mm |
| Input / output gear teeth | 24 / 48 |
| Gear module / pressure angle | 2 mm / 20 degrees |
| Gear center distance / face width | 72 / 6 mm |
| Per-gear tooth-thickness reduction | 0.20 mm |
| Camshaft location, X/Z | -22 / 129.7995 mm |
| Wing hinge locations | X=-17, Y=+/-23, Z=233 mm |
| Wing actuation arm radius | 18 mm |
| Wing eccentricity / fork stroke | 9 / 18 mm |
| Wing angle range | -30 to +30 degrees |
| Jaw hinge | X=55, Y=0, Z=264 mm |
| Jaw rear actuation arm | 24 mm |
| Beak follower stroke | 8.99056 mm |
| Lower bill opening | 0 to 22 degrees |
| Rear jaw counterweight | about 18.82 g at X=-34 mm relative to hinge |
| Beak cam base support radius | 26 mm |

Let phi be the front wheel angle, increasing about +Y. Define the cam phase theta = phi/2; the actual output shaft rotates about +Y by -theta. A complete wing/beak cycle takes **two wheel revolutions**.

| Wheel speed | Pedal cycles per minute | Flaps / beak cycles per minute |
|---|---:|---:|
| 30 rpm | 30 | 15 |
| 60 rpm | 60 | 30 |
| 90 rpm | 90 | 45 |
| 120 rpm | 120 | 60 |

| Cam phase | Wheel angle | Wings | Beak |
|---|---:|---|---|
| 0 degrees | 0 degrees | Down, -30 degrees | Closed |
| 90 degrees | 180 degrees | Level, rising | Begins opening |
| 180 degrees | 360 degrees | Up, +30 degrees | Open, 22 degrees |
| 270 degrees | 540 degrees | Level, falling | Closed |
| 360 degrees | 720 degrees | Down again | Closed |

The speed control changes cycle frequency, not the designed amplitudes, while the gravity-return contacts remain loaded. Speeds above 120 rpm were not checked. Free coasting can slow, stop, and reverse as gravity back-drives the mechanism; it is not a one-way freewheel.

## Derivation

### 1. Legs: a closed linkage that cannot straighten into a toggle

Each ankle is attached to one crank pin, 180 degrees out of phase with the other. The thigh and shin meet at a revolute knee. The hip remains fixed to the torso. Given the instantaneous pedal position P and hip H, the knee is the forward branch of the intersection of circles of radius 68 mm around H and 76 mm around P.

The hip-to-axle distance is

    D = sqrt((38 - (-10))^2 + (90 - 188)^2) = 109.123783 mm

The pedal-to-hip distance d therefore stays in

    D - 22 <= d <= D + 22
    87.123783 <= d <= 131.123783 mm

Two links of lengths 68 and 76 reach continuously when

    |76 - 68| < d < 76 + 68
              8 < d < 144 mm

Both inequalities remain strict throughout the full rotation. The knee bend relative to a straight leg is 48.9087 to 105.7721 degrees, and its minimum absolute sine is 0.753663. Thus it never reaches a straight or fully folded singularity. The feet are fixed to the shins; the ankle rotates around its crank pin, so the feet pitch naturally during the cycle.

The thigh and shin plates occupy different lateral planes, at |Y| = 66 and 72 mm. Knee pins bridge that offset. This prevents the two plates from colliding when they fold.

### 2. Wings: a circular cam and gravity return

Each wing rotates about a longitudinal hinge and has a shoe at an 18 mm outboard arm radius. A vertically translating fork has a shelf underneath each shoe. Raising the outboard shoes raises both wings. Their centers of mass remain outboard, so their weight supplies the return torque.

For cam eccentricity e = 9 mm and a horizontal flat follower,

    q_w(theta) = e (1 - cos(theta))
    alpha(theta) = asin((q_w - e) / 18)
                 = asin(-0.5 cos(theta))

The fork stroke is exactly 18 mm; the wing sweep is exactly -30 to +30 degrees. The contact moment arm never vanishes because cos(alpha) >= cos(30 degrees). The shoes are offset axially from the rotating arm plates so a lifting shelf cannot cut through the arm during rotation.

### 3. Beak: lift behind the hinge; a counterweight closes it

The upper bill stays fixed. A separate guided rod pushes upward on a shoe 24 mm behind the lower jaw hinge. Lifting that rear arm lowers the front tip and opens the mouth.

    beta = asin(q_b / 24)
    H = 24 sin(22 degrees) = 8.990558 mm

The lower bill would not reliably close under its own front-heavy weight. A steel counterweight behind the hinge supplies an explicit closing moment. It is part of the jaw's rigid group and included in the mass/inertia calculation.

The lift law is

    q_b(theta) = H [max(0, -cos(theta))]^3

The beak is closed for half a cam revolution. The cubic power makes displacement, velocity, and acceleration continuous at the opening/dwell boundaries (C2); jerk is not continuous there. Maximum opening coincides with wings-up.

### 4. The shaped cam is an actual flat-follower envelope

Simply using polar radius R + q_b would not yield the desired lift under a flat follower. The tangency point slides sideways. For s(theta) = R + q_b(theta) and R = 26 mm, the local cam contour is

    x(theta) = s sin(theta) + s' cos(theta)
    z(theta) = s cos(theta) - s' sin(theta)

Rotating it about +Y by -theta gives a horizontal tangent at the required height s. The radius of curvature s + s'' stays above 8.018883 mm, so the profile remains convex. The maximum lateral contact excursion is 10.3814 mm, inside the 32 mm wide follower foot. The sampled CAD contour has 960 points, with a measured support-height discretization error of about 0.000205 mm.

These are ideal geometry calculations, not machining-tolerance claims.

## What was executed and checked

### CAD and kinematics

- All 38 exported CAD components are valid B-rep solids/compounds.
- The assembly was checked at 48 poses over two wheel turns. There were 5,751 narrowphase solid-intersection tests and **zero unintended intersections above 0.001 cubic millimetres**. Rigidly grouped parts and the separately checked gear pair are excluded from unintended-pair tests.
- The involute gears were checked at 1,440 ideal-ratio phases. Maximum tooth overlap was zero; minimum surface gap was about 0.188217 mm at the centered phases.
- Leg length closure and cam support/convexity were checked analytically and numerically. Tiny numerical closure errors describe computation, not fabrication accuracy.

Reports: `cad/clearance_report.json`, `cad/gear_check.json`, and `cad/kinematic_checks.json`.

**These are sampled geometric checks, not a continuous swept-volume or tolerance certificate.** Tooth clearance does not validate tooth-contact dynamics.

### Torque-driven custom dynamics

`sim/validate.py` executes six independent coordinates: the wheel angle, fork height, two wing angles, beak slider height, and jaw angle. The exact gear ratio and both leg loops are reduced analytically. The rotating gear/cam masses and the legs' mass, configuration-dependent inertia, and gravitational load are reflected onto the wheel.

A finite-torque velocity servo acts only on the front wheel. Five unilateral contacts drive the fork, wings, and jaw. The active-set impulse solver permits compression only: it cannot pull the followers back down. Gravity, jaw counterweight, and viscous damping produce the return motion. Contact reaction loads are applied back to the wheel, so the input angle is **not prescribed**.

The executed variable-speed test is 16 seconds with a 0.5 ms timestep. It ramps from rest to 30 rpm, then to 90 rpm, then to 45 rpm. Its maximum observed wheel speed was 90.945 rpm. All five contact forces remained positive, with a minimum of about 0.1045 N. The maximum wing and jaw deviations from the ideal geometric solution were below 0.0002 degrees in this simplified model. The movie replays this trace.

Separate steady-speed contact-load calculations at 30, 60, 90, and 120 wheel rpm were positive. At 120 rpm the smallest calculated contact force was approximately 0.0692 N at the jaw shoe. This supports gravity-return operation under the stated model; it does not validate arbitrary accelerations, friction, or higher speeds.

Report and trace: `sim/validation.json`, `sim/contact_trajectory.csv`.

### Interactive browser dynamics

The HTML contains a JavaScript implementation of the same torque-driven reduced contact model, not a lookup animation of joint angles. Tests in headless Chromium exercised 30, 60, 90, and 120 rpm commands for 10-12 simulated seconds, and a six-second motor-off coasting case. States stayed finite, contacts stayed loaded, and no page errors were reported. At 120 rpm the largest reported geometric angular deviation was about 0.000341 degrees.

The headless environment used the Canvas2D rendering fallback. The browser test injected the self-contained HTML content rather than navigating to a file URL because local file navigation was blocked in that environment. The delivered document contains all scripts and mesh data inline.

Report: `sim/browser_validation.json`.

## Articulations and experimental MuJoCo exports

The explicit assembly has **10 revolute joints and 2 prismatic joints**, plus two ankle loop closures in the primary formulation. These are the front wheel, rear wheel, output shaft, two hips, two knees, two wings, jaw, lifting fork, and beak rod. The rear-wheel coordinate is idle and is omitted from the reduced dynamics. The fixed torso and frame are one stationary group.

Three MJCF variants are supplied:

1. `sim/pelican_ideal_gears.xml`: ideal gear-ratio equality, hip/knee articulation trees, and ankle connect constraints. Cam/shoe motion is through contacts.
2. `sim/pelican_contact_gears.xml`: removes the ideal gear equality and uses individual convex tooth colliders. Ankle connect constraints remain.
3. `sim/pelican_all_contact.xml`: **experimental, zero equality constraints**. Gear teeth contact each other, and each pedal pin is captured inside a segmented ankle bore. Hip/knee revolute joints and the remaining ideal guides are retained. The ankle bore uses 24 convex sectors; nominal radial clearance is 0.25 mm, and its polygonal minimum is about 0.2239 mm.

**MuJoCo was not available in this environment, so none of these scenes was loaded or stepped in MuJoCo.** XML parsing and initial ankle-anchor alignment were checked; those are not engine validation. In particular, the all-contact variant may need timestep, contact stiffness, friction, backlash, and ankle-clearance tuning to avoid chatter or disengagement. The executed custom solver does not validate its tooth or ankle contact behavior.

Visual STL meshes are not enabled as general collision geometry. Individual convex teeth, cam shapes, contact shoes, and ankle sectors are used where required. Masses and inertias are supplied explicitly from the CAD rather than inferred from the convex hull of a whole decorative mesh. Guides and frame attachment are idealized joints, not rubbing guide-wall contacts.

To try the exports from this directory:

```bash
python -m pip install mujoco numpy

# Headless, ideal gears and closed ankle constraints
python sim/run_mujoco.py --headless --seconds 16 --rpm 60

# Interactive macOS: use MuJoCo's mjpython launcher
mjpython sim/run_mujoco.py

# Experimental: gear-tooth contacts plus contact-captured pedal pins
mjpython sim/run_mujoco.py --all-contact
```

On Linux the interactive runner can be launched with `python`. The runner ramps the wheel speed rather than instantly imposing it. Revalidate before changing speed beyond the checked range or modifying inertias and friction.

## Rebuild

The existing HTML, video, and STEP need no Python installation to inspect. Rebuilding uses the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python build_cad.py
python check_geometry.py
python sim/validate.py
python sim/export_mjcf.py
python sim/export_contact_ankles.py
python make_viewer.py
python make_media.py
```

`test_viewer.py` additionally requires Playwright and Chromium; its Chromium executable path is environment-specific. Rendering uses VTK offscreen support. `make_media.py` selects DejaVu fonts from common Linux paths for annotations; change those paths on another operating system. No font files are bundled.

Core parameters are in `design.py`. Some frame, guide, and body dimensions remain directly specified in `build_cad.py`; this is inspectable parametric source, not a fully generalized scale-independent configurator. A major parameter change requires updating matching guides, body slots, and collider locations, then rerunning every check.

## Remaining engineering work

This design has not been fabricated. Shaft fastening, keys/set screws, collars, bushings, press fits, fastener access, strength, wear, lubrication, manufacturing tolerances, and physical contact friction are not engineered or validated. The nominal densities used for the mass model are recorded in `cad/manifest.json`; changing materials or weights can change the passive return behavior.

The executed solver uses ideal gears, normal-only contacts, ideal articulated guides, and exact leg-loop reduction. It does not validate Coulomb friction, gear backlash impacts, ankle clearance, joint play, flexible deformation, or engine-specific solvers. The external shape of the lower pouch is rigid; there is no deformable throat simulation.

## Source map

- `design.py`: dimensions, leg geometry, transforms, cam envelope, involute gears.
- `build_cad.py`: actual CadQuery solid construction and exports.
- `cad/manifest.json`: components, rigid groups, nominal material densities, masses, inertias.
- `sim/joint_map.json`: origins, joint axes, parents, gear ratio, ankle anchors.
- `sim/validate.py`, `sim/browser_physics.js`: executed reduced contact models.
- `sim/export_mjcf.py`, `sim/export_contact_ankles.py`: untested engine scene construction.
- `render.py`, `make_media.py`, `make_viewer.py`: actual-CAD visualization.
- `check_geometry.py`, `test_viewer.py`: executed geometric and browser checks.

The mechanism geometry and calculations are original to this prototype. Relevant software behavior is documented at:

- https://cadquery.readthedocs.io/en/latest/importexport.html
- https://mujoco.readthedocs.io/en/stable/XMLreference.html
- https://mujoco.readthedocs.io/en/stable/python.html
