# Pelican bicycle asset

Original CAD mechanism created and supplied by the contributor (liinbor).
Authorship confirmed on 2026-10-20 for inclusion in these Newton examples.
The meshes and MJCF are distributed under Apache-2.0 with this contribution.

Copied from the video-scene archive without geometry changes. The accompanying
README describes the original design and its original validation scope. The
Newton example uses physical gear contact and two closed ankle linkages; only
the front wheel has a motor, controlled by the measured remote fader position.

The two largest visual meshes use indexed binary PLY instead of STL to avoid
duplicating triangle vertices. Triangle positions are unchanged.
