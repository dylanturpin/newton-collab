# Supplied pelican bicycle

Copied from the user's `/Users/liinbor/Downloads/pelican_bicycle/` on
2026-10-14. The STL meshes, source CAD scripts, parameters, original README,
and `sim/pelican_contact_gears.xml` are preserved without geometry edits.
The user's instruction is to retain the entire supplied mechanism, including
its handlebars, and operate its electric drive with a separate physical fader.

`pelican_task.py` adapts the contact MJCF in memory: scales dimensions by 2.5,
assigns explicit contact masks, groups the stand and its linkages into one fixed
articulation, and replaces the source actuator with a finite-torque wheel motor
controlled by the measured fader position. Gear teeth, cams and ankle bearings
remain physical constraints/contacts. Visual triangles use flat normals.
