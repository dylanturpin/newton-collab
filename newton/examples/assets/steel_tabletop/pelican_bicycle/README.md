# Pelican bicycle

Original user-authored CAD meshes and the contact-gear MJCF, in metres.
The Newton example scales the complete mechanism by 2.5 for tabletop use.

The front wheel drives a 24-tooth gear against a 48-tooth gear. The output
shaft turns two cams: one lifts the wing fork, the other opens the lower bill.
Gravity returns the followers. Each leg has hip and knee hinges and a closed
ankle connection to the wheel's pedal crank. The stand remains fixed.

Only the front wheel has a finite-torque velocity motor. Its command comes
from the measured position of the separate, physically movable remote fader.
No wing, jaw, pedal, or fader trajectory is prescribed. Meshes use flat normals.

Run `uv run --extra examples -m newton.examples steel_pelican_bicycle --test --viewer null`
on CUDA to check the complete interaction, gear ratio, ankle closure, wing
and jaw motion, and speed tracking. See `SOURCE.md` for asset provenance.
