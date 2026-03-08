from typing import Union
import math
import numpy as np
import matplotlib.pyplot as plt

from ..core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, EgoFrontBins, StandardDistanceBins, CardinalBinsEgo, CardinalBinsAllo

def relationship_applies(obj1, obj2, relationship, anchor_ori: tuple = (0, 1)) -> bool:
    """Check if relationship applies to obj1 and obj2 from anchor's perspective."""
    p1 = getattr(obj1, 'pos', obj1)
    p2 = getattr(obj2, 'pos', obj2)

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x1 - x2, y1 - y2
    dsq = dx*dx + dy*dy

    ax, ay = float(anchor_ori[0]), float(anchor_ori[1])
    a_len = math.hypot(ax, ay) or 1.0
    axn, ayn = ax / a_len, ay / a_len

    if isinstance(relationship, (PairwiseRelationshipDiscrete)):
        j = relationship.dist.bin_id
        lo, hi = relationship.dist.bin_system.BINS[j]
        dir_bin_id = relationship.direction.bin_id
        bin_system = relationship.direction.bin_system
        d = math.sqrt(dsq)
        if not (d > float(lo) and d < float(hi)):
            return False
        
        # Direction bin check
        # atan2(cross, dot) with normalized anchor; v length cancels out
        dot = axn*dx + ayn*dy
        cross = axn*dy - ayn*dx
        deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
        bid, _ = bin_system.bin(deg)
        return bid == dir_bin_id

    if isinstance(relationship, PairwiseRelationship):
        has_dir = relationship.direction is not None
        has_dist = relationship.dist is not None

        # Distance check when needed
        if has_dist:
            target_d = float(getattr(relationship.dist, 'value', 0.0))
            d = math.sqrt(dsq)
            if abs(d - target_d) > 1e-6:
                return False

        if has_dir:
            # Only degree comparison is needed now
            dot = axn*dx + ayn*dy
            cross = axn*dy - ayn*dx
            deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
            if abs(deg - float(relationship.degree)) > 1e-6:
                return False
        return True

    if isinstance(relationship, ProximityRelationship):
        th = float(getattr(relationship, 'PROXIMITY_THRESHOLD', 5.0))
        if not (dsq <= th * th + 1e-6):
            return False
        # Must match the discrete pairwise inside the proximity relation
        return relationship_applies(p1, p2, relationship.pairwise_rel, anchor_ori)

    raise ValueError(f"Invalid relationship type: {type(relationship)}")


# ---- domain generator ----
def generate_points_for_relationship(
    anchor_pos: tuple,
    relationship: Union[PairwiseRelationship, PairwiseRelationshipDiscrete],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    anchor_ori: tuple[int, int] = (0, 1),
) -> set[tuple[int, int]]:
    """
    Generate integer points (x,y) within ranges that satisfy the relationship
    with the anchor_pos. The anchor_pos is treated as obj2 by default
    (i.e., we test relationship_applies(candidate, anchor_pos, ...)).

    Notes:
    - Only Pairwise/PairwiseDiscrete supported.
    - Handles distance-only or distance+degree. Degree-only not supported.

    TODO debug
    """
    ax, ay = int(anchor_pos[0]), int(anchor_pos[1])
    xmin, xmax = int(x_range[0]), int(x_range[1])
    ymin, ymax = int(y_range[0]), int(y_range[1])

    out: set[tuple[int, int]] = set()

    # ---- Pairwise / PairwiseDiscrete ----
    # Determine distance window [Rmin, Rmax]
    Rmin, Rmax = 0.0, None
    if isinstance(relationship, PairwiseRelationshipDiscrete) and relationship.dist is not None:
        j = relationship.dist.bin_id
        if j is not None and relationship.dist.bin_system is not None:
            lo, hi = relationship.dist.bin_system.BINS[j]
            # same-distance bin detected by j==0
            if j == 0:
                return out
            Rmin, Rmax = float(lo), float(hi)
    elif isinstance(relationship, PairwiseRelationship) and relationship.dist is not None:
        d = float(relationship.dist.value)
        Rmin, Rmax = max(0.0, d - 1e-6), d + 1e-6

    # If no distance bound, we do not generate (degree-only not supported)
    if Rmax is None:
        return out

    # x scan with y ranges from circle ring
    X0 = max(xmin, int(math.ceil(ax - Rmax + 1e-9)))
    X1 = min(xmax, int(math.floor(ax + Rmax - 1e-9)))
    Rmax2, Rmin2 = float(Rmax * Rmax), float(max(Rmin, 0.0) * max(Rmin, 0.0))

    # Precompute for fast discrete checks
    aox, aoy = float(anchor_ori[0]), float(anchor_ori[1])
    alen = math.hypot(aox, aoy) or 1.0
    aoxn, aoyn = aox/alen, aoy/alen

    is_disc = isinstance(relationship, PairwiseRelationshipDiscrete)
    disc_dir_bin = None
    disc_bin_system = None
    if is_disc:
        disc_dir_bin = relationship.direction.bin_id
        disc_bin_system = relationship.direction.bin_system or EgoFrontBins()

    for x in range(X0, X1 + 1):
        dx = float(x - ax)
        t2 = Rmax2 - dx*dx
        if t2 < 0: continue
        yspan = math.sqrt(t2) if t2 > 0.0 else 0.0
        y_top = int(math.floor(ay + yspan - 1e-9))
        y_bot = int(math.ceil(ay - yspan + 1e-9))
        for y in range(max(ymin, y_bot), min(ymax, y_top) + 1):
            if x == ax and y == ay:
                continue
            dy = float(y - ay)
            dsq = dx*dx + dy*dy
            if not (dsq > Rmin2 and dsq < Rmax2):
                continue
            if is_disc:
                dot = aoxn*dx + aoyn*dy
                cross = aoxn*dy - aoyn*dx
                deg = -math.degrees(math.atan2(cross, dot)) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0
                bid, _ = disc_bin_system.bin(deg)
                if bid == disc_dir_bin:
                    out.add((x, y))
            else:
                if relationship_applies((x, y), (ax, ay), relationship, anchor_ori):
                    out.add((x, y))
    return out



# ------------ rose glyph for paper plot ------------

# def rose_glyph_pretty(
#     pairs,                      # list of (sector_id, ring_id)
#     color="#E6A23C",
#     ring_colors=("#E8F3FF", "#EAF7F0", "#FFF3E0", "#F3E8FF"),
#     tick_colors=("#4C78A8","#8E9ED6","#74C0E3","#BDE0FE","#A8DADC","#95D5B2","#FFD166","#F4A261"),
#     gap=0.92,
#     n_dirs=8,
#     n_rings=4,
#     figsize=(2.4, 2.4),
#     ring_heights=(0.4, 0.25, 0.25, 0.25),
#     bg_band_ratio=1,          # 0..1, portion of eacF4A261h ring colored near the outer edge
#     sector_highlight_color="#FDE68A",
#     sector_highlight_alpha=0.9
# ):
#     """Rose glyph with (sector, ring) fills; background rings colored on outer band, and selected sectors highlighted."""
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
#     ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
#     ax.set_xticks([]); ax.set_yticks([]); ax.set_rlim(0, 1)

#     # radial boundaries
#     rh = np.array(ring_heights[:n_rings], dtype=float); rh /= rh.sum()
#     rb = np.concatenate([[0.0], np.cumsum(rh)])

#     sector_angle = 2*np.pi / n_dirs
#     width = sector_angle * gap
#     sectors = {int(s) % n_dirs for s, _ in pairs}

#     # background rings: only outer band portion
#     for i, c in enumerate(ring_colors[:n_rings]):
#         h = rb[i+1] - rb[i]
#         band_h = h * max(0.0, min(1.0, bg_band_ratio))
#         bottom = rb[i+1] - band_h
#         ax.bar(0, band_h, width=2*np.pi, bottom=bottom, color=c, alpha=0.5, linewidth=0)

#     # sector highlights (any sector appearing in `pairs`)
#     for s in sectors:
#         theta = s * sector_angle
#         ax.bar(theta, 1.0, width=width, bottom=0.0, color=sector_highlight_color,
#                linewidth=0, alpha=sector_highlight_alpha, align='center')

#     # target (sector, ring) wedges
#     theta_map = {s: s * sector_angle for s in sectors}
#     for s, r in pairs:
#         s = int(s) % n_dirs; r = int(r)
#         if 0 <= r < n_rings:
#             theta = theta_map.get(s, s * sector_angle)
#             ax.bar(theta, rb[r+1]-rb[r], width=width, bottom=rb[r],
#                    align='center', linewidth=0, color=color)

#     # ring separators
#     th = np.linspace(0, 2*np.pi, 361)
#     for r in rb[1:-1]: ax.plot(th, np.full_like(th, r), lw=1.1, color="white")

#     # ticks
#     centers = np.arange(n_dirs) * sector_angle
#     for i, th_c in enumerate(centers):
#         ax.plot([th_c, th_c], [1.02, 1.08], lw=2.4,
#                 color=tick_colors[i % len(tick_colors)], solid_capstyle='round')

#     for sp in ax.spines.values(): sp.set_visible(False)
#     ax.set_rlim(0, 1.10)
#     return fig, ax

import numpy as np
import matplotlib.pyplot as plt

def rose_glyph_compass_style(
    pairs,                      # [(sector_id, ring_id)]
    color="#2563EB",            # still used for selected cells
    gap=0.92, n_dirs=8, n_rings=4,
    figsize=(2.6, 2.6),
    ring_heights=(0.40, 0.25, 0.25, 0.25),
    sector_highlight_color=None,
    # --- new styling knobs ---
    ring_edge_colors=None,      # dopamine palette for ring borders
    edge_frac_min=0.10,         # thinnest border (as fraction of that ring's thickness)
    edge_frac_max=0.20,         # thickest border (as fraction of that ring's thickness)
    edge_alpha=0.10,            # VERY faint when not selected
    edge_alpha_selected=0.95,   # strong when selected
    sector_highlight_alpha=0.28 # stronger sector highlight
):
    """
    Compass-styled rose with ring borders as inner-edge bands only.
    - No background fills.
    - Sector highlight retained (stronger).
    - Ring border colors from a dopamine palette; borders sit INSIDE each ring.
    """
    if sector_highlight_color is None:
        sector_highlight_color = color

    # 多巴胺风格的单色系（蓝色）调色盘：高饱和、由浅到深
    if ring_edge_colors is None:
        ring_edge_colors = [
            "#7DD3FC",  # sky blue (light)
            "#38BDF8",  # vivid sky
            "#0EA5E9",  # bright azure
            "#0284C7",  # strong azure
            "#0369A1",  # deep cyan-blue
            "#3B82F6",  # bright blue
            "#2563EB",  # royal blue
            "#1D4ED8",  # deep royal
        ]

    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=figsize)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_rlim(0, 1)

    # geometry
    rh = np.array(ring_heights[:n_rings], float); rh /= rh.sum()
    rb = np.r_[0.0, rh.cumsum()]                     # ring boundaries in [0,1]
    ang = 2*np.pi/n_dirs; width = ang*gap
    sectors = {int(s) % n_dirs for s, _ in pairs}
    selected_rings = {int(r) for _, r in pairs if 0 <= int(r) < n_rings}

    # 1) no background bands

    # 2) stronger sector highlight
    for s in sectors:
        ax.bar(s*ang, 1.0, width=width, bottom=0,
               color=sector_highlight_color, alpha=sector_highlight_alpha, lw=0, zorder=1)

    # 3) target cells (unchanged)
    for s, r in pairs:
        s = int(s) % n_dirs; r = int(r)
        if 0 <= r < n_rings:
            ax.bar(s*ang, rb[r+1]-rb[r], width=width, bottom=rb[r],
                   color=color, edgecolor="white", lw=0.8, align='center', zorder=5)

    # 4) ring borders: thin bands INSIDE each ring (no outside bleed, no gaps)
    #    inner ring thinner; outer rings thicker (linear ramp)
    edge_fracs = np.linspace(edge_frac_min, edge_frac_max, n_rings)
    for i in range(n_rings):
        r_in, r_out = rb[i], rb[i+1]
        h = r_out - r_in
        band_h = h * float(edge_fracs[i])           # strictly within [r_out - band_h, r_out]
        band_bottom = r_out - band_h                # e.g., ... 0.8–1.0, 1.8–2.0, ...
        alpha = edge_alpha_selected if i in selected_rings else edge_alpha
        c = ring_edge_colors[i % len(ring_edge_colors)]
        ax.bar(0, band_h, width=2*np.pi, bottom=band_bottom,
               color=c, alpha=alpha, lw=0, zorder=6)

    # 5) no extra outline strokes (prevents perceived gaps)

    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_rlim(0, 1.12)
    return fig, ax

if __name__ == "__main__":
    # relationship = PairwiseRelationshipDiscrete.relationship((4, 6), (0, 0), anchor_ori=(1, 0), bin_system=CardinalBinsEgo(), distance_bin_system=StandardDistanceBins())
    # points = generate_points_for_relationship((0, 0), relationship, (-20, 20), (-20, 20), (1, 0))
    # for p in sorted(points):
    #     dist = math.hypot(p[0] - 0.0, p[1] - 0.0)
    #     print(f"Point {p}: distance = {dist:.2f}")

    # sector: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    # ring: near=0, mid=1, slightly far=2, far=3
    all_pairs = [(i, j) for i in range(8) for j in range(4)]
    # ax = rose_glyph_pretty(pairs=[(7, 1), (6, 1), (7, 2), (6, 2)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(7, 2), (0, 1), (7, 1)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(6, 1), (5, 2), (5, 1), (6, 2)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(6, 3), (7, 2), (6, 2), (7, 3)], color="#E59E1B")
    # ax = rose_glyph_pretty(pairs=[(3, 2)], color="#E59E1B")


    # ax = rose_glyph_compass_style(pairs=[(6, 1), (6, 2), (7, 1), (7, 2)])
    # ax = rose_glyph_compass_style(pairs=[(6, 1), (6, 2), (7, 2)])
    # ax = rose_glyph_compass_style(pairs=[(5, 1), (4, 0), (4, 1), (3, 1)])
    # ax = rose_glyph_compass_style(pairs=[(3, 2)])
    # ax = rose_glyph_compass_style(pairs=[(4, 1)])
    # ax = rose_glyph_compass_style(pairs=[(5, 1), (5, 2), (6, 2)])
    # ax = rose_glyph_compass_style(pairs=[(6, 1), (6, 2), (7, 2)])
    # ax = rose_glyph_compass_style(pairs=all_pairs)
    ax = rose_glyph_compass_style(pairs=[(1, 1), (2, 2), (1, 2)])

    plt.savefig("rose_glyph_pretty.png", transparent=True)