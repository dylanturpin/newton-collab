# Contact-driven bird automaton

A dimensioned CAD prototype with one powered crank, a three-gear train, and two passive outputs: mirrored wing flapping and opening of the lower beak. The torso is fixed. All modeled mechanical components are rigid.

**Start here:** open `media/bird_contact_motion.mp4` and `media/mechanism_overview.png`. Import `cad/bird_automaton.step` to inspect the actual solid assembly. Individual STEP and STL parts, the parametric CadQuery source, simulator scenes, and validation scripts are included.

## What is implemented and what is tested

This is actual B-rep CAD, not an image-generated schematic. Fifteen CAD components export as valid solids/compounds. Eight moving joint coordinates describe the assembly: six revolute joints and two prismatic joints. The cams are rigidly attached to the output shaft, and the counterweight is rigidly attached to the lower jaw.

The following checks were executed in this environment:

* Analytic contact-load calculations over 36,001 cam phases using CAD-derived masses and inertias.
* A 10-second, six-cycle simulation of the five passive coordinates under gravity and unilateral normal contact, with a 0.25 ms time step. Cam rotation is prescribed through ideal gear ratios. The source is `sim/validate.py`; the movie replays its simulated joint coordinates, not keyframed wing/jaw angles.
* Sampled CAD interference tests at 24 shaft phases, 15 degrees apart. No unintended intersection exceeding 0.001 cubic millimetres was found. This is a sampled test, not a continuous swept-volume certificate.
* Planar gear checks at 1,440 shaft phases using the generated involute profiles. No tooth overlap was found at the centered, ideal gear phases. Minimum gaps were about 0.188 mm. This checks geometry, not tooth-contact dynamics.

**MuJoCo was not installed and could not be downloaded in this environment.** The two MJCF scenes and runner are supplied, but they have not been loaded or stepped in MuJoCo here. Treat them as an untested engine integration, not an engine-validation result. The contact-gears scene is the next validation step.

The CAD is a simulation/kinematic prototype, not a fabrication drawing set. Keyways, shaft collars, gear fastening, bearing hardware, strength, wear, friction, and production tolerances remain to be engineered.

## Mechanism and force path

```
24-tooth input gear -> 20-tooth idler -> 36-tooth output gear / shaft
                                         |
                    +--------------------+---------------------+
                    |                                          |
          eccentric circular cam                      beak dwell cam
                    | normal contact                           | normal contact
          guided lifting fork                          guided pushrod
                    |                                          |
         two outboard wing shoes                   shoe BEHIND the jaw hinge
                    |                                          |
          two hinged wings rise                     lower beak tip moves DOWN
                    |                                          |
           gravity lowers wings                    rear counterweight closes jaw
```

Only the input crank has an actuator. There are no wing targets, jaw targets, springs, tendons, four-bar loops, or hidden return motors. Each contact transmits compression only. Rounded contact shoes are integral sliding surfaces, not freely rotating rollers.

### Wing arrangement

Each wing has a fixed longitudinal hinge. Its external crank arm carries a round shoe 18 mm outboard of that hinge. The translating fork has one shelf below each shoe. Raising the fork pushes both outboard shoes upward, so both wings rise. Their mass acts outboard of the hinges and lowers them as the fork descends.

This sign is important: pushing an **inboard** tab would move the wing tip in the opposite direction. This design pushes the **outboard** arm.

The contact shoes are offset axially from the wing arms. The shelves occupy X = -52 to -44 mm, while the crank-arm plate occupies X = -42 to -38 mm. That 2 mm separation prevents the shelf from cutting into the rotating arm. The wing sheet is farther forward. The wing sheet also starts 10 mm outboard of its hinge, clearing the stationary bearing brackets.

The fork is one rigid part with one prismatic coordinate. Its two upright rails enter fixed square guide collars; the simulation represents those guides with an ideal prismatic joint rather than guide-wall contact.

### Beak arrangement

The lower jaw rotates about a lateral hinge at X = 50 mm, Z = 218 mm. A rear arm extends behind the hinge, with its lifting shoe at an 18 mm radius. The pushrod lifts this rear shoe. The front tip therefore moves downward and the mouth opens.

An approximately 8.9 g steel counterweight, centered 28 mm behind the hinge, supplies the closing torque. Without a rear counterweight or spring, the lower jaw could instead remain open under gravity; this return force is explicitly modeled.

The jaw shoe is shifted sideways: the lifting shelf is at Y = -12 to -4 mm, clear of the main jaw arm at Y = -2 to +2 mm. A 0-degree jaw stop establishes the closed position. The fixed upper beak has clearance from the lower jaw.

## Dimensions and intended motion

All CAD dimensions below are in **millimetres**. MJCF uses **metres**, kilograms, and seconds.

| Feature | Value |
|---|---:|
| Base | 170 x 246 x 10 |
| Maximum fixed head height | 239 |
| Input / idler / output teeth | 24 / 20 / 36 |
| Gear module / pressure angle | 2 / 20 degrees |
| Gear face width | 8 |
| Gear center distances | 44, 56 |
| Tooth thickness reduction | 0.20 per gear; approximately 0.40 pair tangential backlash |
| Camshaft height | 65 |
| Wing hinge height / lateral positions | 190 / Y = +/-22 |
| Wing shoe arm radius | 18 |
| Wing cam radius / eccentricity | 20 / 10.3243758543 |
| Fork total lift | 20.6487517086 |
| Wing angle range | -35 to +35 degrees |
| Beak cam base support radius | 18 |
| Beak follower maximum lift | 9 |
| Rear jaw shoe radius from hinge | 18 |
| Jaw angle range | 0 to 30 degrees |
| Beak counterweight | 8.895 g, 28 mm behind hinge |
| Input speed / output speed | 54 rpm / 36 rpm |
| Flap rate | 0.6 Hz |

The input and output turn in the same direction because the idler introduces the second reversal. The idler changes direction and spacing, not the final tooth-count ratio.

| Output shaft phase | Wings | Beak |
|---|---:|---:|
| 0 degrees | -35 degrees, down | Closed |
| 90 degrees | Level, rising | Closed; begins opening |
| 180 degrees | +35 degrees, up | 30 degrees open |
| 270 degrees | Level, falling | Closed |
| 360 degrees | -35 degrees, down | Closed |

The beak remains closed over the half-revolution extending from 270 degrees through 0 degrees to 90 degrees.

## Derivation

Let theta be output-shaft angle, increasing from the wings-down position.

### Circular eccentric cam and wings

For eccentricity e, the center of the wing cam follows

    Yc = e sin(theta)
    Zc = Zshaft - e cos(theta)

A horizontal flat follower contacts the top of this circular cam. Its displacement above the lowest position is exactly

    q_w(theta) = e [1 - cos(theta)]

For a wing shoe at radius r from its hinge,

    r sin(alpha) = q_w - e
    alpha(theta) = asin[-(e/r) cos(theta)]

Choosing r = 18 and e = r sin(35 degrees) gives exactly -35 to +35 degrees of wing travel and 2e = 20.64875 mm of fork travel. This is a geometric result, not an animation assumption.

The wing contact stays well away from a toggle: cos(alpha) is at least cos(35 degrees), approximately 0.819. Contact lever arms therefore remain nonzero throughout the stroke.

### Jaw and closed-mouth dwell

For a rear jaw arm of radius a, an upward displacement q_b gives

    beta = asin(q_b/a)

With a = 18 and q_b,max = 9, the maximum opening is 30 degrees.

The beak lift law is

    q_b(theta) = 9 cos(theta)^2       for pi/2 < theta < 3pi/2
                 0                   otherwise

This has a 180-degree closed-mouth dwell and zero velocity at the rise/dwell transitions. It is C1 but not C2: acceleration changes discontinuously at those transitions. It is adequate for the tested slow motion; a smoother cam law and revised envelope would be appropriate for faster operation.

### The actual flat-follower cam profile

A naive polar radius R + q(theta) would NOT generally produce the requested lift under a flat follower. Contact slides sideways as the cam turns. This CAD instead uses the support-function envelope.

Let

    s(theta) = R + q_b(theta)
    s'(theta) = dq_b/dtheta

The local YZ contour is

    Y(theta) = s sin(theta) + s' cos(theta)
    Z(theta) = s cos(theta) - s' sin(theta)

When the cam rotates by theta, that surface has the required horizontal tangent at height R + q_b. The tangency point moves laterally by s'. Its maximum lateral excursion is 9 mm, within the 32 mm wide follower foot.

The local radius of curvature is s + s''. With R = 18, its minimum is 9 mm. Thus the profile stays convex and does not require an undercut or a non-convex collider. The STEP profile samples this envelope with 720 points.

For the circular wing cam, the contact point moves laterally by at most 10.3244 mm; its follower foot spans +/-18 mm.

### Return-force check

A passive follower is valid only while the required normal forces stay nonnegative. Using the CAD-derived inertias, the wing contact force is calculated from

    N_w r cos(alpha) = I_w alpha_ddot + b_w alpha_dot + gravity restoring torque

and the jaw contact force from

    N_b a cos(beta) = I_b beta_ddot + b_b beta_dot - gravity jaw torque

The rear counterweight makes the jaw gravity torque closing over the entire range. Slider masses and vertical accelerations are then included to obtain the cam normal loads.

At 36 output rpm, the analytic minima were approximately 0.149 N at each wing shoe, 0.388 N at the wing cam, 0.129 N at the jaw shoe, and 0.172 N at the beak cam. They remain positive, supporting gravity-return operation under the modeled conditions. Peak follower-load torque at the output shaft is approximately 0.0051 N m; that excludes gear/bearing friction and motor acceleration.

Exact machine-readable numbers and assumptions are in `sim/validation.json`.

## Simulation models

### Validated reduced-coordinate model

`sim/validate.py` integrates five passive coordinates: fork Z, left wing angle, right wing angle, beak-slider Z, and jaw angle. Each is an independent ideal prismatic or revolute articulation to the fixed frame. Gravity and viscous joint damping are applied from the CAD-derived masses and inertias. A projected impulse solver enforces nonpenetrating, compression-only contacts and joint stops.

Only the rotating cams have prescribed motion. Their normal-contact boundaries come from the derived geometric support functions. The three gears use ideal ratios; their tooth forces are not included in this validation model. Contacts are normal-only, with no Coulomb sliding friction.

The 10-second test used a 0.00025 s time step. Wing and jaw angle errors relative to the geometric solution were below 0.001 degree after the first second. This measures numerical agreement in the simplified model, NOT physical manufacturing accuracy or an independently validated frictional mechanism.

### Exported MuJoCo models -- not engine-tested here

`sim/bird_ideal_gears.xml` uses two gear-ratio equality constraints. The five wing/jaw drive contacts remain actual contact pairs.

`sim/bird_contact_gears.xml` removes those equality constraints and uses per-tooth convex colliders plus root cylinders. The generated individual tooth polygons were checked to be convex. It has only revolute/prismatic articulations, contact pairs, joint stops, and the single crank velocity actuator.

Neither scene enables the full visual STL meshes as general colliders. Whole-gear convex hulls would fill the tooth gaps. The frame and guide surfaces are represented by ideal joints; their incidental collisions are not enabled. The CAD interference check is a separate safeguard, not part of the engine model.

The MJCF body inertias are explicitly calculated from the CAD volumes rather than inferred from convex visual-mesh hulls.

## Run

From this directory:

```bash
# Run the exported MuJoCo scene (engine integration still needs verification)
python -m pip install mujoco numpy
python sim/run_mujoco.py --headless --seconds 10

# macOS interactive viewer: use MuJoCo's mjpython launcher
mjpython sim/run_mujoco.py

# Experimental tooth-contact version
mjpython sim/run_mujoco.py --contact-gears
```

On Linux, `python sim/run_mujoco.py` can launch the interactive viewer. The runner ramps crank speed to 54 rpm over 0.6 s, and provides `--rpm` to change it. The gravity-return contact-load check was performed at 54 input rpm / 36 output rpm only; other speeds should be revalidated.

To rebuild and rerun the included validation:

```bash
python -m pip install cadquery numpy shapely vtk Pillow 'imageio[ffmpeg]'
python build_cad.py
python sim/validate.py
python check_clearance.py
python check_gears.py
python sim/export_mjcf.py
python make_media.py
```

The base parameters live in `design.py`. Some bracket dimensions remain directly specified in `build_cad.py`, so changing overall scale, shoulder locations, or shaft layout requires updating the matching brackets and collision geometry too. The source is inspectable CAD, not a fully generalized design configurator.

## Files

* `cad/bird_automaton.step`: full neutral-format solid assembly in the wings-down pose.
* `cad/*.step`, `cad/*.stl`: individual local-coordinate CAD components.
* `cad/manifest.json`: local component origins, axes, and grouping/motion metadata.
* `media/mechanism_overview.png`: annotated actual-CAD view and derived linkage sections.
* `media/bird_contact_motion.mp4`: replay of the custom contact simulation.
* `sim/contact_trajectory.csv`, `sim/validation.json`: simulated states and numerical checks.
* `cad/clearance_report.json`, `cad/gear_clearance.json`: executed geometry checks.
* `design.py`, `build_cad.py`, `render.py`, `make_media.py`: reproducible design and rendering source.
* `sim/*.xml`, `sim/run_mujoco.py`, `sim/export_mjcf.py`: untested MuJoCo integration and exporter.

## External implementation references

The geometry and mechanism derivation above are original calculations for this prototype. Relevant software behavior is documented by the primary sources below:

* CadQuery assembly and STEP export: https://cadquery.readthedocs.io/en/stable/importexport.html
* MuJoCo MJCF, explicit contacts, joints, and mesh convex-hull behavior: https://mujoco.readthedocs.io/en/stable/XMLreference.html
* MuJoCo Python viewer and macOS mjpython launcher: https://mujoco.readthedocs.io/en/stable/python.html
