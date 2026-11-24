"""
Murri Oki karaoke trophy generator.

Builds a multi-part, colour-swappable trophy:
  base.stl            - stepped plinth with a locating peg and a plate pocket (print black)
  hand.stl            - forearm + fist gripping the mic (print black, gold, silver, bronze...)
  mic_handle.stl      - mic handle; slides down through the fist (print in accent colour)
  mic_head.stl        - grille ball head; pegs onto the handle (accent colour)
  plate_1st/2nd/3rd   - name plates that press into the base pocket
  plate_blank.stl     - plain plate to add your own text in Bambu Studio

Requirements: pip install manifold3d trimesh numpy scikit-image matplotlib
Run:          python3 generate_trophy.py
All units mm.
"""

import os
import numpy as np
import trimesh
from manifold3d import Manifold, CrossSection, FillRule, Mesh
from skimage.measure import marching_cubes
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl")
os.makedirs(OUT, exist_ok=True)

SEG = 96              # circle resolution for mechanical parts
CLR = 0.3             # fit clearance (per side) for sliding joints
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"

# ---------------------------------------------------------------- dimensions
MIC_R = 9.0           # mic handle radius where it passes through the fist
PEG = 12.0            # square locating peg (base -> forearm)
PEG_H = 10.0

BASE_W, BASE_D = 72.0, 56.0      # main block
BASE_H = 40.0
PLINTH_W, PLINTH_D, PLINTH_H = 88.0, 70.0, 9.0
CAP_W, CAP_D, CAP_H = 76.0, 60.0, 4.0
PLATE_W, PLATE_H, PLATE_T = 62.0, 30.0, 1.6     # pocket size (plate is CLR smaller)
TEXT_T = 0.8                                    # raised text height on plate


# ---------------------------------------------------------------- helpers
def to_manifold(tm: trimesh.Trimesh) -> Manifold:
    mesh = Mesh(vert_properties=np.asarray(tm.vertices, dtype=np.float32),
                tri_verts=np.asarray(tm.faces, dtype=np.uint32))
    m = Manifold(mesh)
    assert m.status().name == "NoError", m.status()
    return m


def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mm = m.to_mesh()
    return trimesh.Trimesh(vertices=mm.vert_properties[:, :3], faces=mm.tri_verts, process=True)


def save(m: Manifold, name: str):
    tm = to_trimesh(m)
    path = os.path.join(OUT, name)
    tm.export(path)
    bb = tm.bounds
    print(f"{name:20s} {tm.is_watertight=}  size={np.round(bb[1]-bb[0],1)}  tris={len(tm.faces)}")
    return tm


def rounded_box(w, d, h, r):
    """Box sitting on z=0, centred in xy, vertical edges rounded with radius r."""
    sq = CrossSection.square([w - 2 * r, d - 2 * r], center=True).offset(r, join_type=__import__("manifold3d").JoinType.Round, circular_segments=48)
    return Manifold.extrude(sq, h)


def chamfer_slab(w, d, h, c, r):
    """Rounded slab with a 45-degree chamfer of size c around its top edge."""
    body = rounded_box(w, d, h - c, r)
    top = Manifold.hull(rounded_box(w, d, 0.01, r).translate([0, 0, h - c]) +
                        rounded_box(w - 2 * c, d - 2 * c, 0.01, max(r - c, 0.5)).translate([0, 0, h - 0.01]))
    return body + top


def text_section(txt, size):
    """Text as a CrossSection, centred at the origin, cap height ~= size."""
    tp = TextPath((0, 0), txt, size=size * 1.37, prop=FontProperties(fname=FONT))
    polys = [p for p in tp.to_polygons() if len(p) >= 3]
    cs = CrossSection(polys, FillRule.NonZero)
    (x0, y0), (x1, y1) = np.array(cs.bounds()).reshape(2, 2)
    return cs.translate([-(x0 + x1) / 2, -(y0 + y1) / 2])


def rot_to(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def affine(R, t):
    return np.hstack([R, np.asarray(t, float).reshape(3, 1)])


# ---------------------------------------------------------------- SDF hand
def sd_capsule(p, a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    d = np.linalg.norm(pa - h[:, None] * ba, axis=1)
    return d - (ra + (rb - ra) * h)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


# Fist local frame: mic bar runs along +Z through the origin (index finger on top),
# back of the hand toward +Y, palm/finger-fronts face -Y (the viewer).
FINGERS = [  # (z level, finger radius, size scale)
    (18.0, 6.3, 1.00),   # index
    (5.6, 6.5, 1.03),    # middle
    (-7.0, 6.2, 0.98),   # ring
    (-18.2, 5.4, 0.90),  # pinky
]
WRIST_L = np.array([-34.0, 7.0, -3.0])
FOREARM_TILT = np.radians(38)      # forearm leaves the fist angled toward the pinky side


def hand_local_sdf(p):
    d = np.full(len(p), 1e9)
    # palm / metacarpals (back of hand)
    for z, r, s in FINGERS:
        d = smin(d, sd_capsule(p, [-26, 9, z * 0.55 - 2], [5 * s, 12.5 * s, z], 8.0, r + 0.9), 4.0)
    # meaty palm under the bar + heel of the hand
    d = smin(d, sd_capsule(p, [-28, 2, -12], [-13, -1, 4], 11.0, 9.0), 5.0)
    d = smin(d, sd_capsule(p, [-34, 6, -10], [-34, 6, 8], 14.0, 13.0), 6.0)
    # curled fingers: MCP -> PIP -> DIP -> tip, wrapping the bar
    fingers = np.full(len(p), 1e9)
    for z, r, s in FINGERS:
        mcp = np.array([8.0, 12.5, z]) * [s, s, 1]
        pip = np.array([16.5, -3.0, z]) * [s, s, 1]
        dip = np.array([4.0, -16.5, z]) * [s, s, 1]
        tip = np.array([-8.5, -12.5, z]) * [s, s, 1]
        f = sd_capsule(p, mcp, pip, r + 0.6, r + 0.2)
        f = smin(f, sd_capsule(p, pip, dip, r + 0.3, r), 1.5)
        f = smin(f, sd_capsule(p, dip, tip, r, r - 0.6), 1.5)
        fingers = smin(fingers, f, 0.9)          # small k keeps the creases between fingers
    d = smin(d, fingers, 2.5)
    # thumb: thenar pad -> wraps across the front of index/middle
    zi = FINGERS[0][0]
    thumb = sd_capsule(p, [-24, -2, zi - 8], [-12, -22, zi - 3], 9.0, 7.0)
    thumb = smin(thumb, sd_capsule(p, [-12, -22, zi - 3], [6, -24.5, zi - 5.5], 7.0, 6.0), 2.0)
    d = smin(d, thumb, 2.2)
    return d


def build_hand_and_mic():
    forearm_dir_local = np.array([-np.cos(FOREARM_TILT), 0, -np.sin(FOREARM_TILT)])
    R = rot_to(forearm_dir_local, np.array([0, 0, -1.0]))     # local -> world
    bar_world = R @ np.array([0, 0, 1.0])
    if bar_world[0] < 0:          # make the mic point up-and-to-the-right
        M = np.diag([-1, 1, 1.0])
        R = M @ R
        bar_world = R @ np.array([0, 0, 1.0])

    FOREARM_LEN = 72.0
    wrist_w = np.array([0, 0, FOREARM_LEN])
    origin_w = wrist_w - R @ WRIST_L                           # world position of the fist origin
    Rinv = R.T

    # grid in world space
    lo = np.array([-55, -45, -3.0])
    hi = np.array([55, 45, origin_w[2] + 42])
    step = 0.4
    axes = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    d = np.empty(len(G))
    CH = 2_000_000
    for i in range(0, len(G), CH):
        g = G[i:i + CH]
        pl = (g - origin_w) @ Rinv.T                           # world -> local
        dh = hand_local_sdf(pl)
        # forearm: elliptical round cone, flatter front-to-back
        gs = g * [1, 1 / 0.78, 1]
        df = sd_capsule(gs, [0, 0, -8], wrist_w + [0, 0, 6], 15.5, 17.5)
        dd = smin(dh, df, 7.0)
        dd = np.maximum(dd, -g[:, 2])                          # flat bottom at z=0
        d[i:i + CH] = dd
    vol = d.reshape([len(a) for a in axes])
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(step,) * 3)
    verts += lo
    tm = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(tm, iterations=8)
    if tm.volume < 0:
        tm.invert()
    hand = to_manifold(tm).simplify(0.02)

    # ---- mic (built along +Z, then placed on the bar axis)
    to_bar = affine(rot_to(np.array([0, 0, 1.0]), bar_world), origin_w)
    s_tail, s_collar, s_flare, s_top = -46.0, 27.0, 42.0, 50.0
    profile = [  # (s, radius) handle outline
        (s_tail, 0.0), (s_tail, 7.2), (s_tail + 1.2, 8.4), (s_tail + 4, MIC_R),
        (s_collar, MIC_R), (s_flare, 12.6), (s_flare + 0.01, 13.4), (s_top - 2.6, 13.4),
        (s_top - 2.6, 12.0), (s_top, 11.4), (s_top, 0.0)]
    pts = np.array([[r, s] for s, r in profile])
    handle = Manifold.revolve(CrossSection([pts]), circular_segments=SEG)
    # small tail ring detail + rounded peg for the head
    handle = handle - Manifold.revolve(CrossSection.square([0.8, 1.2]).translate([MIC_R - 0.6, s_tail + 6]), SEG)
    head_peg_r, head_peg_h = 5.0, 8.0
    handle = handle + Manifold.cylinder(head_peg_h, head_peg_r, head_peg_r - 0.4, SEG).translate([0, 0, s_top])

    HEAD_R = 19.5
    head_c = s_top + 15.5                                       # head sphere centre on the axis
    cut = head_c - s_top                                        # flat underside distance below centre
    head = Manifold.sphere(HEAD_R, 128)
    # mesh grille: criss-cross grooves in a 1.1 mm shell
    shell = Manifold.sphere(HEAD_R + 1, 128) - Manifold.sphere(HEAD_R - 1.1, 128)
    slabs = []
    pitch, gw = 3.3, 1.05
    for k in np.arange(-HEAD_R, HEAD_R + pitch, pitch):
        slabs.append(Manifold.cube([gw, 2 * HEAD_R + 4, 2 * HEAD_R + 4], center=True).translate([k, 0, 0]))
        slabs.append(Manifold.cube([2 * HEAD_R + 4, gw, 2 * HEAD_R + 4], center=True).translate([0, k, 0]))
    grid = Manifold.batch_boolean(slabs, __import__("manifold3d").OpType.Add)
    keep_low = Manifold.cube([60, 60, 10], center=True).translate([0, 0, -cut - 5 + 4.5])  # solid skirt at bottom
    head = head - ((grid ^ shell) - keep_low)
    # equator band (where a real mic's grille halves join)
    band = Manifold.cylinder(5.0, HEAD_R + 0.6, HEAD_R + 0.6, 128, center=True).translate([0, 0, -3])
    head = head + (band ^ Manifold.sphere(HEAD_R + 0.6, 128).scale([1, 1, 3]))
    head = head.trim_by_plane([0, 0, 1], -cut)                  # flat bottom (prints with no support)
    head = head - Manifold.cylinder(head_peg_h + 0.4, head_peg_r + 0.15, head_peg_r + 0.15, SEG).translate([0, 0, -cut - 0.01])
    head_placed = head.translate([0, 0, head_c])

    # ---- holes / joints in the hand
    hole = Manifold.cylinder(200, MIC_R + CLR, MIC_R + CLR, SEG).translate([0, 0, -100])
    hand = hand - hole.transform(to_bar)
    # collar seat so the mic flare rests snugly on the index finger/thumb
    seat = Manifold.revolve(CrossSection([np.array([[0, s_collar], [MIC_R + CLR, s_collar],
                                                    [12.6 + CLR, s_flare], [40, s_flare], [40, 90], [0, 90]])]), SEG)
    hand = hand - seat.transform(to_bar)
    sock = Manifold.cube([PEG + 2 * CLR, PEG + 2 * CLR, PEG_H + 0.6], center=True).translate([0, 0, (PEG_H + 0.6) / 2 - 0.01])
    hand = hand - sock

    handle_w = handle.transform(to_bar)
    head_w = head_placed.transform(to_bar)
    return hand, handle, head, handle_w, head_w, bar_world


def build_base(arm_xy):
    base = chamfer_slab(PLINTH_W, PLINTH_D, PLINTH_H, 2.5, 3.0)
    base += rounded_box(BASE_W, BASE_D, BASE_H, 2.0).translate([0, 0, PLINTH_H - 0.01])
    top_z = PLINTH_H + BASE_H
    cap = chamfer_slab(CAP_W, CAP_D, CAP_H, 1.2, 3.0).translate([0, 0, top_z - 0.01])
    base += cap
    top_z += CAP_H
    # decorative groove around the block
    groove = rounded_box(BASE_W + 2, BASE_D + 2, 1.2, 3).translate([0, 0, PLINTH_H + 5]) - rounded_box(BASE_W - 1.6, BASE_D - 1.6, 2, 1.2).translate([0, 0, PLINTH_H + 4.6])
    base -= groove
    # front plate pocket
    pz = PLINTH_H + 5 + 1.2 + (BASE_H - 6.2) / 2 + 0.5
    pocket = Manifold.cube([PLATE_W, PLATE_T * 2, PLATE_H], center=True).translate([0, -BASE_D / 2, pz])
    base -= pocket
    # locating peg (chamfered tip) under the forearm
    peg = Manifold.cube([PEG, PEG, PEG_H - 1.0], center=True).translate([0, 0, (PEG_H - 1) / 2])
    peg += Manifold.hull(Manifold.cube([PEG, PEG, 0.01], center=True).translate([0, 0, PEG_H - 1]) +
                         Manifold.cube([PEG - 2, PEG - 2, 0.01], center=True).translate([0, 0, PEG_H]))
    base += peg.translate([arm_xy[0], arm_xy[1], top_z - 0.01])
    return base, top_z


def build_plate_lines(lines):
    """Plate with any number of centred lines: [(text, cap_height_mm, gap_below_mm), ...].
    Lines too wide for the plate are squeezed horizontally to fit."""
    w, h = PLATE_W - 2 * CLR, PLATE_H - 2 * CLR
    plate = rounded_box(w, h, PLATE_T, 1.0)
    raised = rounded_box(w - 2.0, h - 2.0, TEXT_T, 1.0) - rounded_box(w - 4.0, h - 4.0, TEXT_T + 1, 0.6).translate([0, 0, -0.5])
    total = sum(s for _, s, _ in lines) + sum(g for _, _, g in lines[:-1])
    y = total / 2
    for txt, size, gap in lines:
        t = text_section(txt, size)
        (x0, _), (x1, _) = np.array(t.bounds()).reshape(2, 2)
        if x1 - x0 > w - 8:
            t = t.scale([(w - 8) / (x1 - x0), 1])
        raised += Manifold.extrude(t, TEXT_T).translate([0, y - size / 2, 0])
        y -= size + gap
    return plate + raised.translate([0, 0, PLATE_T - 0.01])


def build_plate_parts(lines):
    """Same layout as build_plate_lines, but returned as separate pieces
    {name: Manifold} sharing one coordinate system, so each can take its own
    colour in Bambu Studio (import together as one multi-part object)."""
    w, h = PLATE_W - 2 * CLR, PLATE_H - 2 * CLR
    z = PLATE_T - 0.01
    parts = {"plate": rounded_box(w, h, PLATE_T, 1.0),
             "border": (rounded_box(w - 2.0, h - 2.0, TEXT_T, 1.0) -
                        rounded_box(w - 4.0, h - 4.0, TEXT_T + 1, 0.6).translate([0, 0, -0.5])).translate([0, 0, z])}
    total = sum(s for _, s, _ in lines) + sum(g for _, _, g in lines[:-1])
    y = total / 2
    for i, (txt, size, gap) in enumerate(lines, 1):
        t = text_section(txt, size)
        (x0, _), (x1, _) = np.array(t.bounds()).reshape(2, 2)
        if x1 - x0 > w - 8:
            t = t.scale([(w - 8) / (x1 - x0), 1])
        slug = "".join(c if c.isalnum() else "_" for c in txt.lower()).strip("_")
        parts[f"line{i}_{slug}"] = Manifold.extrude(t, TEXT_T).translate([0, y - size / 2, z])
        y -= size + gap
    return parts


CHAMPION_LINES = [("CHAMPION", 7.5, 3.2), ("MURRI-OKI", 5.0, 2.8), ("2026 MEMBERS CONFERENCE", 2.8, 0)]


def build_plate(line1, line2):
    w, h = PLATE_W - 2 * CLR, PLATE_H - 2 * CLR
    plate = rounded_box(w, h, PLATE_T, 1.0)
    frame = rounded_box(w - 2.0, h - 2.0, TEXT_T, 1.0) - rounded_box(w - 4.0, h - 4.0, TEXT_T + 1, 0.6).translate([0, 0, -0.5])
    raised = frame
    if line1:
        t1 = text_section(line1, 8.0)
        (x0, _), (x1, _) = np.array(t1.bounds()).reshape(2, 2)
        if x1 - x0 > w - 9:
            t1 = t1.scale([(w - 9) / (x1 - x0), 1])
        raised += Manifold.extrude(t1, TEXT_T).translate([0, 4.2 if line2 else 0, 0])
    if line2:
        t2 = text_section(line2, 5.2)
        raised += Manifold.extrude(t2, TEXT_T).translate([0, -7.0, 0])
    return plate + raised.translate([0, 0, PLATE_T - 0.01])


if __name__ == "__main__":
    hand, handle, head, handle_w, head_w, bar = build_hand_and_mic()

    # put the base peg under the forearm, and centre the base under the whole
    # assembly's centre of mass so it can't tip forward
    assembly = hand + handle_w + head_w
    com = np.array(to_trimesh(assembly).center_mass)
    arm_xy = -com[:2] * 0.85
    arm_xy = np.clip(arm_xy, [-BASE_W / 2 + 20, -BASE_D / 2 + 18], [BASE_W / 2 - 20, BASE_D / 2 - 18])
    print("centre of mass offset", np.round(com, 1), "-> arm at", np.round(arm_xy, 1))
    base, top_z = build_base(arm_xy)

    save(base, "base.stl")
    save(hand, "hand.stl")
    save(handle.translate([0, 0, -handle.bounding_box()[2]]), "mic_handle.stl")   # stands on its tail
    save(head.translate([0, 0, 15.5]), "mic_head.stl")
    plates = {"1st": ("MURRI OKI", "1ST PLACE"), "2nd": ("MURRI OKI", "2ND PLACE"),
              "3rd": ("MURRI OKI", "3RD PLACE"), "blank": (None, None)}
    for k, (a, b) in plates.items():
        save(build_plate(a, b), f"plate_{k}.stl")
    save(build_plate_lines(CHAMPION_LINES), "plate_champion.stl")
    os.makedirs(os.path.join(OUT, "plate_champion_parts"), exist_ok=True)
    for k, m in build_plate_parts(CHAMPION_LINES).items():
        save(m, f"plate_champion_parts/champion_{k}.stl")

    # full assembly for previewing (not for printing)
    off = [arm_xy[0], arm_xy[1], top_z]
    parts = {
        "base": base,
        "hand": hand.translate(off),
        "mic_handle": handle_w.translate(off),
        "mic_head": head_w.translate(off),
        "plate": build_plate(*plates["1st"]).rotate([90, 0, 0]).translate(
            [0, -BASE_D / 2 + PLATE_T, PLINTH_H + 5 + 1.2 + (BASE_H - 6.2) / 2 + 0.5]),
    }
    scene = trimesh.Scene()
    for k, m in parts.items():
        scene.add_geometry(to_trimesh(m), node_name=k, geom_name=k)
    scene.export(os.path.join(OUT, "assembly_preview.3mf"))
    for k, m in parts.items():
        to_trimesh(m).export(os.path.join(OUT, f"_asm_{k}.stl"))
    print("total height", round(to_trimesh(Manifold.batch_boolean(list(parts.values()), __import__('manifold3d').OpType.Add)).bounds[1][2], 1))
