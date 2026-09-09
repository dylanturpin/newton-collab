.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

:orphan:

FeatherPGS torsional-grip diagnosis
=================================

The small pinch reproduces the suspected mechanism on ``fpgs-main`` commit
``16f98c3cf3979f0e6f1732b40117f7a444216ae4``. A sphere and a capsule gripped at
its rounded ends can rotate almost freely about the line joining the two
contacts, despite sufficient squeeze force and tangential friction. This is
consistent with missing torsional friction. It does not establish the cause of
an unmeasured production grasp whose contact geometry or spin axis differs.

No production solver or collision code is changed by this diagnostic.

Controlled experiment
---------------------

Two prismatic jaws each push inward with 10 N along X. They grip a 1 kg object
with a 20 mm radius or box half-width. Capsule cylindrical half-lengths are
30 mm and, in a separate short-capsule control, 3 mm. All objects use the same
isotropic inertia, 0.00016 kg m², to isolate geometry from mass-distribution and
gyroscopic differences. This is a controlled diagnostic inertia, not the
natural inertia of every shape.

After 0.3 s of settling, apply 0.002 N m about X for 0.5 s, using a 0.005 s
timestep. Tangential friction is 0.8 and the solver uses 64 iterations. Gravity,
rolling friction, and angular damping are zero. The force-controlled jaws avoid
inferring normal load from an artificially overlapping pair of fixed pads.
FeatherPGS's measured total normal load remains approximately 20 N.

With no resisting torque, the predicted final angular speed is
``torque * duration / inertia = 6.25 rad/s``.

CPU results
-----------

Measured with Warp ``1.17.0.dev20260807`` on macOS ARM, FeatherPGS ``split`` mode:

.. list-table:: Final spin about the pinch axis
   :header-rows: 1
   :widths: 42 14 20 24

   * - Case
     - Contacts
     - Torsional coefficient
     - Angular speed [rad/s]
   * - FeatherPGS sphere
     - 2
     - 0
     - 6.2500
   * - FeatherPGS capsule ends
     - 2
     - 0
     - 6.2466
   * - FeatherPGS capsule sides
     - 4
     - 0
     - less than 0.000001
   * - FeatherPGS box faces
     - 8
     - 0
     - less than 0.000001
   * - FeatherPGS sphere, friction 5 and 256 iterations
     - 2
     - 0
     - 6.2500
   * - FeatherPGS sphere
     - 2
     - 0.005 m
     - 6.2500
   * - XPBD sphere
     - 2
     - 0
     - 6.2501
   * - XPBD sphere
     - 2
     - 0.005 m
     - 0
   * - XPBD capsule ends
     - 2
     - 0
     - 6.2463
   * - XPBD capsule ends
     - 2
     - 0.005 m
     - 0

The short capsule side case also retains four contacts and resists the torque.
Its final speed is 0.000016 rad/s.
There is no evidence of the linked implementation's capsule contact-deduplication
failure in these Newton cases. Capsule-box pairs take Newton's GJK/MPR path;
finding an analytic capsule-box helper in the source does not establish that
the collision pipeline calls it.

CUDA verification
-----------------

The same twelve-case matrix was repeated on an NVIDIA RTX A6000 with Warp
``1.16.0``, CUDA 12.9, and FeatherPGS ``matrix_free`` mode. The sphere reaches
6.2500 rad/s and capsule ends reach 6.2434 rad/s. Box faces and long capsule
sides remain below 0.000001 rad/s; short capsule sides reach 0.000012 rad/s.
The contact counts are unchanged: two for sphere/capsule ends, four for capsule
sides, and eight for box faces. Increasing friction/iterations or setting
FeatherPGS's torsional material coefficient still does not stop sphere spin.
XPBD's torsional-material control holds both sphere and capsule ends at zero
spin while preserving their two geometric contacts.

Why the missing mode matters
----------------------------

FeatherPGS constructs one normal and two tangent force rows at each witness.
Their angular Jacobians are ``r cross direction``. For the sphere's contacts at
``r = (+/-R, 0, 0)``, every point-friction torque has zero X component. Increasing
friction or the number of PGS iterations cannot supply a missing angular row.
The measured tangent angular Jacobian has rank two, with X as its null direction.

The capsule-end GJK witnesses are slightly off-axis numerically, so their angular
Jacobian is technically full-rank. Its spin-Jacobian column has a norm of only
about 0.08 mm in this fixture and provides negligible resistance: the body
reaches nearly the free spin prediction. Capsule side contacts and box face contacts have substantial
separation perpendicular to X and can resist that torque through ordinary
point friction.

``ModelBuilder.ShapeConfig.mu_torsional`` already exists, with length units and
a default of 0.005 m. ``Model.shape_material_mu_torsional`` carries it, but
FeatherPGS does not read that array. XPBD does: its angular friction correction is
bounded by normal load times the torsional coefficient. Changing only that
coefficient in XPBD stops sphere and capsule-end spin without adding witnesses.
This is the positive control for the missing material response.

The `referenced AVBD change
<https://github.com/LuckyIYI/avbd-metal/commit/a5da1c7f61318f29f23556d2c27a9d6df30f65a1>`_
adds the same type of bounded contact-normal torque as an aggregate manifold
mode. It also fixes a separate capsule witness deduplication issue. Those are
distinct mechanisms; this experiment supports the torsional-friction mechanism
for the two-point pinch.

Implications for an implementation
---------------------------------

A finite contact area can be represented by a signed angular constraint along
the patch normal, with zero linear Jacobian and an impulse limit
``abs(lambda_twist) <= mu_torsional * sum(lambda_normal)``. With 20 N total squeeze
and 0.005 m, the torque budget would be 0.1 N m, comfortably above this test's
0.002 N m. The existing material field supplies the coefficient.

This should be integrated into the solver's row allocation, all supported solve
paths, warm-start/reset handling, and diagnostics. Aggregate the normal load
once per compatible contact region so witness count does not multiply torque
capacity. Do not manufacture a ring of sphere contacts to obtain a moment arm.
Patch grouping or positional anchors at the same two geometric locations do
not, by themselves, add the missing twist mode. Rotation about a tangent axis is
a separate rolling-friction question.

Reproduce
---------

Run the focused tests from the repository root:

.. code-block:: console

   uv run --extra dev python -m unittest -v newton.tests.test_feather_pgs_torsional_grip

Four tests should pass. The fifth,
``test_feather_pgs_honors_torsional_material``, is explicitly an expected failure:
it requests less than 0.01 rad/s and observes about 6.25 rad/s. This records the
missing behavior; it does not claim a fix. Once torsion is implemented, remove
that expected-failure annotation.

Print all measured cases and save their geometry and load data:

.. code-block:: console

   uv run --extra dev python -m newton.tests.test_feather_pgs_torsional_grip \
       --probe --matrix --device cpu --output /tmp/fpgs-torsion-cpu.json
   uv run --extra dev python -m newton.tests.test_feather_pgs_torsional_grip \
       --probe --matrix --device cuda:0 --mode matrix_free --output /tmp/fpgs-torsion-cuda.json

For one case, omit ``--matrix`` and use ``--shape sphere``, ``--shape capsule_end``,
``--shape capsule_side``, or ``--shape box``. ``--solver xpbd`` and
``--mu-torsional 0.005`` select the positive control. The test suite uses CPU when
CUDA is unavailable and CUDA otherwise; the probe explicitly chooses its device.
