# Pelican bicycle

Original CAD mechanism created and supplied by liinbor, contributed under
Apache-2.0. Converted to a self-contained binary USD for this example; no
external asset download, STL loading, or MJCF conversion is needed at runtime.
The original CAD/MJCF source is retained in this branch's history at commit
`9231afb9`, under `newton/examples/assets/steel_tabletop/pelican_bicycle/`.

The USD embeds the visual meshes and convex collision geometry at the tabletop
scale (2.5 times the original mechanism). Authored masses, inertias, joint
limits, contact filtering, and both closed ankle linkages are preserved.
The front wheel is the only motorized mechanism joint; its speed comes from
the measured position of the separate physical remote fader.
