# Pin-tumbler padlock

Run `python -m newton.examples steel_padlock` in the standard viewer. The **Cutaway**
checkbox switches the housing and plug display meshes; the complete collision geometry
stays active. Use the normal viewer pause, single-step, camera and reset controls, or
hold/scrub the **Key path** to inspect insertion and turning. `--no-cutaway` shows the
complete padlock; `--key wrong` substitutes one incorrectly cut key.

The housing is held in a fixture with gravity disabled. Bounded six-DOF joint drives
insert and turn the independently moving key, solving drive impulses together with
contacts rather than applying a timestep-dependent external PD force. The turning
phase waits until the key seats. Five independently moving key pins and five spring-loaded
drivers constrain the passive plug. Its actuator releases two locking balls and the
spring opens the retained shackle. No key-code comparison or time-based rule unlocks it.
The drives are limited to 3 N total translational force and 8 mN·m axial turning torque.
Key translation and rotation also have velocity limits.

The 60 Hz example uses four substeps and 32 Steel sweeps per substep, for a 1/240 s
physics timestep. Sixteen sweeps remain insufficient for this tightly coupled mechanism.
Contact search distances account for timestep, insertion speed and initial shackle
acceleration; collision surfaces and clearances are unchanged. Passive springs use
fixed-rest implicit drives. Slider armature conditions the small pins and locking
balls against the heavier shackle, giving a minimum effective slider inertia of
one sixteenth of the shackle assembly's mass. This numerical inertia changes
transients, not body masses or spring equilibria. Internally, lengths are mm,
mass is kg and inertia is kg·mm² to condition the small keyway clearances. A complete run
is 216 frames (3.6 simulated seconds). CUDA is required.

Startup warms one physics substep before capturing the complete frame, without
eagerly loading unrelated CUDA modules. First-use Warp kernel compilation can
still take longer than reopening the example with an existing kernel cache.

`padlock.json.gz` is one frozen numeric specimen, with SI geometry, exact convex cells,
mass/inertia, passive joint definitions, visual meshes and corner normals. Python's
standard `gzip`/`json` readers load it; no CAD package, private repository or asset download
is needed. Display meshes have zero density and no collision flags. The matching lock
has 791 collision cells, including 472 in the housing and 54 in the key. Cutaway cap
faces and custom normals are baked into separate display meshes. Flat housing panels
keep face normals; the round shackle uses radial normals away from its machined seats.
Display colors are encoded as sRGB for the viewer.

The geometry was authored parametrically for this example; no third-party CAD meshes
are included. It uses a generic asymmetric three-groove blank and five cuts (`24132`),
scaled by 0.8. Source specimen fingerprint:
`be726a5e0221485f7b454d2267d0dd27ecbdc39dd017856b567f5adc7085f2b8`.
The bundled packet omits unused keys and adds display-only section meshes.

Design references:

- [Schlage cylinder service manual](https://commercial.schlage.com/content/dam/allegion-us-2/web-files/schlage/technical-documents/Schlage_Everest_Full_Size_Cylinder_Service_Manual_108484.pdf): pin pitch, cut depths and pin-length/shear-line dimensions.
- [American Lock A3650 service drawing](https://cdn.masterlock.com/masterlock/resources/documents/pdf/A3650_SeriesTechnicalServiceManualPage.pdf): opposed locking balls, actuator and retained shackle.

This is one reference mechanism. Friction and spring values are simulation assumptions;
hardware force calibration, key withdrawal and relocking are outside its validation.
