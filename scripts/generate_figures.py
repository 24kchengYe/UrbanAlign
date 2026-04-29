"""
Generate publication-quality PDF figures for UrbanAlign main paper.
Figure 1: Conceptual comparison of three approaches
Figure 2: Framework overview (3-stage pipeline)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ─── Global style (Arial) ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'mathtext.fontset': 'dejavusans',
})

# ─── Colors ─────────────────────────────────────────────────────
C_PIPE_A  = '#6C7A89'
C_PIPE_B  = '#8E9EAB'
C_OURS    = '#1A5276'
C_ACCENT  = '#16A085'
C_BAD     = '#C0392B'
C_GOOD    = '#27AE60'
C_CLIP    = '#16A085'
C_FEED    = '#A93226'

ECCV_DIR = r"D:\pythonPycharms\工具开发\052AI4SyntheticData\ECCV_2026_Paper_Template"


def draw_rounded_box(ax, xy, w, h, color, text, fontsize=8,
                     linewidth=1.0, fill_alpha=0.08, text_color='black',
                     bold=False, zorder=2, linestyle='-', va_offset=0):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=(*matplotlib.colors.to_rgb(color), fill_alpha),
                         edgecolor=color, linewidth=linewidth,
                         linestyle=linestyle, zorder=zorder)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2 + va_offset, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, zorder=zorder+1,
            linespacing=1.3, family='Arial')
    return box


def draw_arrow(ax, start, end, color='black', linewidth=1.2,
               style='->', zorder=1, linestyle='-'):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        linewidth=linewidth, color=color, linestyle=linestyle, zorder=zorder)
    ax.add_patch(arrow)
    return arrow


def draw_cross(ax, cx, cy, size=0.15, color=C_BAD, linewidth=2.5):
    ax.plot([cx - size, cx + size], [cy - size, cy + size],
            color=color, linewidth=linewidth, solid_capstyle='round', zorder=6)
    ax.plot([cx - size, cx + size], [cy + size, cy - size],
            color=color, linewidth=linewidth, solid_capstyle='round', zorder=6)


def draw_check(ax, cx, cy, size=0.18, color=C_GOOD, linewidth=2.5):
    ax.plot([cx - size, cx - size*0.25, cx + size],
            [cy, cy - size*0.7, cy + size*0.6],
            color=color, linewidth=linewidth, solid_capstyle='round',
            solid_joinstyle='round', zorder=6)


def draw_urban_scene(ax, cx, cy, scale=1.0):
    """Draw a person observing an urban street scene."""
    s = scale

    # ── Sky background ──
    sky = FancyBboxPatch((cx - 0.65*s, cy - 0.15*s), 1.3*s, 1.15*s,
                          boxstyle="round,pad=0.02",
                          facecolor='#E8F4FD', edgecolor='none', zorder=3)
    ax.add_patch(sky)

    # ── Buildings ──
    bldg_color = '#7F8C8D'
    bldg_light = '#95A5A6'
    bldg_dark = '#5D6D7E'

    # Building 1 (left, tall)
    b1 = FancyBboxPatch((cx - 0.55*s, cy + 0.05*s), 0.28*s, 0.75*s,
                          boxstyle="round,pad=0.01",
                          facecolor=bldg_light, edgecolor=bldg_dark,
                          linewidth=0.6, zorder=4)
    ax.add_patch(b1)
    # Windows for building 1
    for wy in [0.2, 0.4, 0.6]:
        for wx in [0.06, 0.16]:
            win = plt.Rectangle((cx + (-0.55 + wx)*s, cy + (0.05 + wy)*s),
                                0.06*s, 0.08*s,
                                facecolor='#F9E79F', edgecolor=bldg_dark,
                                linewidth=0.3, zorder=5)
            ax.add_patch(win)

    # Building 2 (center, medium)
    b2 = FancyBboxPatch((cx - 0.22*s, cy + 0.05*s), 0.30*s, 0.55*s,
                          boxstyle="round,pad=0.01",
                          facecolor=bldg_color, edgecolor=bldg_dark,
                          linewidth=0.6, zorder=4)
    ax.add_patch(b2)
    for wy in [0.15, 0.35]:
        for wx in [0.05, 0.17]:
            win = plt.Rectangle((cx + (-0.22 + wx)*s, cy + (0.05 + wy)*s),
                                0.07*s, 0.09*s,
                                facecolor='#AED6F1', edgecolor=bldg_dark,
                                linewidth=0.3, zorder=5)
            ax.add_patch(win)

    # Building 3 (right, short)
    b3 = FancyBboxPatch((cx + 0.13*s, cy + 0.05*s), 0.25*s, 0.40*s,
                          boxstyle="round,pad=0.01",
                          facecolor='#ABB2B9', edgecolor=bldg_dark,
                          linewidth=0.6, zorder=4)
    ax.add_patch(b3)
    for wy in [0.12, 0.28]:
        win = plt.Rectangle((cx + 0.19*s, cy + (0.05 + wy)*s),
                             0.08*s, 0.06*s,
                             facecolor='#D5F5E3', edgecolor=bldg_dark,
                             linewidth=0.3, zorder=5)
        ax.add_patch(win)

    # Tree
    trunk = plt.Rectangle((cx + 0.42*s, cy + 0.05*s), 0.04*s, 0.18*s,
                            facecolor='#6E4B35', edgecolor='none', zorder=4)
    ax.add_patch(trunk)
    foliage = plt.Circle((cx + 0.44*s, cy + 0.30*s), 0.10*s,
                          facecolor='#27AE60', edgecolor='#1E8449',
                          linewidth=0.5, zorder=5)
    ax.add_patch(foliage)

    # ── Road / ground ──
    road = plt.Rectangle((cx - 0.65*s, cy - 0.15*s), 1.3*s, 0.20*s,
                           facecolor='#D5D8DC', edgecolor='none', zorder=3.5)
    ax.add_patch(road)

    # ── Person silhouette (left side, looking right) ──
    person_x = cx - 0.48*s
    person_y = cy - 0.05*s

    # Head
    head = plt.Circle((person_x, person_y + 0.38*s), 0.06*s,
                       facecolor='#2C3E50', edgecolor='none', zorder=6)
    ax.add_patch(head)

    # Body (triangle)
    body_x = [person_x - 0.06*s, person_x + 0.06*s, person_x]
    body_y = [person_y + 0.05*s, person_y + 0.05*s, person_y + 0.32*s]
    ax.fill(body_x, body_y, color='#2C3E50', zorder=6)

    # Perception lines (emanating from person towards buildings)
    for angle_deg in [-10, 5, 20]:
        angle = np.radians(angle_deg)
        dx = 0.4 * s * np.cos(angle)
        dy = 0.4 * s * np.sin(angle)
        ax.plot([person_x + 0.08*s, person_x + 0.08*s + dx],
                [person_y + 0.30*s, person_y + 0.30*s + dy],
                color=C_OURS, linewidth=0.6, linestyle='--',
                alpha=0.5, zorder=5.5)

    # Eye icon next to person
    eye_x = person_x + 0.12*s
    eye_y = person_y + 0.38*s
    eye = matplotlib.patches.Ellipse((eye_x, eye_y), 0.10*s, 0.05*s,
                                      facecolor='white', edgecolor=C_OURS,
                                      linewidth=0.6, zorder=7)
    ax.add_patch(eye)
    pupil = plt.Circle((eye_x + 0.01*s, eye_y), 0.015*s,
                        facecolor=C_OURS, edgecolor='none', zorder=8)
    ax.add_patch(pupil)


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: Conceptual comparison
# ═══════════════════════════════════════════════════════════════
def create_figure1():
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 2.6))
    ax.set_xlim(-0.3, 14.8)
    ax.set_ylim(-0.2, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')

    row_a = 3.15
    row_b = 2.0
    row_c = 0.85

    bw, bh = 2.2, 0.70
    gap = 0.30
    x_start = 2.5

    # ── Input box (left) — urban scene with person ──
    inp_x, inp_w, inp_h = 0.0, 1.7, 3.0
    inp_y = row_b - inp_h/2 + 0.05
    box_inp = FancyBboxPatch((inp_x, inp_y), inp_w, inp_h,
                              boxstyle="round,pad=0.06",
                              facecolor='#FAFAFA', edgecolor='#AAAAAA',
                              linewidth=0.8, zorder=2)
    ax.add_patch(box_inp)

    # Draw urban scene inside the box
    scene_cx = inp_x + inp_w / 2
    scene_cy = inp_y + inp_h * 0.55
    draw_urban_scene(ax, scene_cx, scene_cy, scale=1.15)

    # Label below scene
    ax.text(inp_x + inp_w/2, inp_y + 0.18,
            'Human Perception\nof Urban Environment',
            ha='center', va='center', fontsize=6.5, color='#444',
            fontstyle='italic', zorder=9, family='Arial')

    # ── Helper: draw one pipeline row ──
    def draw_pipeline_row(y_center, color, box_texts, result_text,
                          verdict_ok, verdict_text, label_char,
                          lw=1.0, fill_a=0.08, result_fontsize=7):
        y = y_center - bh/2
        boxes_x = []
        for i, txt in enumerate(box_texts):
            x = x_start + i * (bw + gap)
            boxes_x.append(x)
            draw_rounded_box(ax, (x, y), bw, bh, color, txt,
                             fontsize=8, linewidth=lw, fill_alpha=fill_a)
        res_w = 2.5
        res_x = boxes_x[-1] + bw + gap
        draw_rounded_box(ax, (res_x, y), res_w, bh, color, result_text,
                         fontsize=result_fontsize, linewidth=lw, fill_alpha=fill_a)

        draw_arrow(ax, (inp_x + inp_w, y_center),
                   (boxes_x[0], y_center), color=color, linewidth=lw)
        for i in range(len(boxes_x) - 1):
            draw_arrow(ax, (boxes_x[i] + bw, y_center),
                       (boxes_x[i+1], y_center), color=color, linewidth=lw)
        draw_arrow(ax, (boxes_x[-1] + bw, y_center),
                   (res_x, y_center), color=color, linewidth=lw)

        vx = res_x + res_w + 0.35
        if verdict_ok:
            draw_check(ax, vx, y_center)
        else:
            draw_cross(ax, vx, y_center)
        vc = C_GOOD if verdict_ok else C_BAD
        ax.text(vx + 0.4, y_center, verdict_text, fontsize=7,
                color=vc, ha='left', va='center', zorder=5,
                linespacing=1.2, family='Arial')

        ax.text(x_start - 0.12, y + bh + 0.06, f'({label_char})',
                fontsize=7, fontstyle='italic', color='#999', zorder=5,
                family='Arial')
        return res_x, res_w

    # (a) End-to-End VLM
    draw_pipeline_row(row_a, C_PIPE_A,
                      ['End-to-End\nZero-Shot VLM', '?\nBlack Box\nProcessing'],
                      '"Image A looks\nmore wealthy"\n(No explanation)',
                      False, 'Low\nInterpretability', 'a')

    # (b) Low-Level
    draw_pipeline_row(row_b, C_PIPE_B,
                      ['Low-Level\nFeature Extraction', 'Pixel Counting\nSegmentation\n(DeepLab / NDVI)'],
                      'Greenery: 30%\nBuilding: 45%\nAcc. 59.1%',
                      False, 'Limited\nSemantic Depth', 'b')

    # (c) UrbanAlign — custom with dimension bars
    y_c = row_c - bh/2
    lw_c = 1.5

    x0 = x_start
    draw_rounded_box(ax, (x0, y_c), bw, bh, C_OURS,
                     'UrbanAlign\n(Ours)', fontsize=9, linewidth=lw_c,
                     fill_alpha=0.12, bold=True, text_color=C_OURS)
    x1 = x0 + bw + gap
    draw_rounded_box(ax, (x1, y_c), bw, bh, C_OURS,
                     'Mid-Level\nSemantic Decoding\nVLM + Multi-Agent',
                     fontsize=6.5, linewidth=lw_c, fill_alpha=0.12,
                     text_color=C_OURS)

    res_w = 2.5
    res_x = x1 + bw + gap
    box_res = FancyBboxPatch((res_x, y_c), res_w, bh,
                              boxstyle="round,pad=0.05",
                              facecolor=(*matplotlib.colors.to_rgb(C_OURS), 0.12),
                              edgecolor=C_OURS, linewidth=lw_c, zorder=2)
    ax.add_patch(box_res)

    bar_data = [('Facade', 8.2, 0.82), ('Pavement', 6.5, 0.65),
                ('Cleanliness', 4.2, 0.42)]
    bar_y_top = y_c + bh - 0.10
    bar_h = 0.055
    bar_spacing = 0.155
    max_bar_w = 0.9
    label_x = res_x + 0.08
    bar_x = res_x + 0.95

    for i, (name, score, frac) in enumerate(bar_data):
        by = bar_y_top - i * bar_spacing
        ax.text(label_x, by, f'{name}:', fontsize=5.5, va='center',
                color='#444', zorder=5, fontweight='medium', family='Arial')
        bar = FancyBboxPatch((bar_x, by - bar_h/2), max_bar_w * frac, bar_h,
                              boxstyle="round,pad=0.008",
                              facecolor=C_ACCENT, edgecolor='none', zorder=5)
        ax.add_patch(bar)
        ax.text(bar_x + max_bar_w * frac + 0.06, by, f'{score}',
                fontsize=5.5, va='center', color='#333', zorder=5, family='Arial')

    ax.text(res_x + res_w/2, y_c + 0.05, 'Acc. 72.3%',
            fontsize=6.5, fontweight='bold', ha='center', va='center',
            color=C_OURS, zorder=5, family='Arial')

    draw_arrow(ax, (inp_x + inp_w, row_c), (x0, row_c),
               color=C_OURS, linewidth=lw_c)
    draw_arrow(ax, (x0 + bw, row_c), (x1, row_c),
               color=C_OURS, linewidth=lw_c)
    draw_arrow(ax, (x1 + bw, row_c), (res_x, row_c),
               color=C_OURS, linewidth=lw_c)

    vx = res_x + res_w + 0.35
    draw_check(ax, vx, row_c)
    ax.text(vx + 0.4, row_c, 'Accurate &\nInterpretable', fontsize=7,
            color=C_GOOD, ha='left', va='center', zorder=5,
            linespacing=1.2, family='Arial')

    ax.text(x_start - 0.12, y_c + bh + 0.06, '(c)',
            fontsize=7, fontstyle='italic', color='#999', zorder=5,
            family='Arial')

    fig.tight_layout(pad=0.15)
    return fig


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: Framework overview (3-stage pipeline)
# ═══════════════════════════════════════════════════════════════
def create_figure2():
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    ax.set_xlim(-0.3, 14.0)
    ax.set_ylim(-0.8, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Right edge for all content
    R_EDGE = 13.6
    bh = 0.75
    arr_lw = 1.2

    # ── Stage background bands ──
    band_w = R_EDGE + 0.2 - (-0.2)  # = 14.0
    bands = [
        (7.4, 9.25),
        (3.9, 7.15),
        (-0.3, 3.65),
    ]
    for y_bot, y_top in bands:
        bg = FancyBboxPatch((-0.2, y_bot), band_w, y_top - y_bot,
                             boxstyle="round,pad=0.1",
                             facecolor=(*matplotlib.colors.to_rgb(C_OURS), 0.03),
                             edgecolor=(*matplotlib.colors.to_rgb(C_OURS), 0.18),
                             linewidth=0.7, zorder=0)
        ax.add_patch(bg)

    # ════════════════════════════════
    #  STAGE 1
    # ════════════════════════════════
    s1_y = 8.0

    draw_rounded_box(ax, (0, s1_y - 0.05), 2.8, 0.9, C_OURS,
                     'Stage 1\nSemantic Dimension\nExtraction',
                     fontsize=8, linewidth=1.2, fill_alpha=0.15,
                     text_color=C_OURS, bold=True)

    draw_rounded_box(ax, (3.5, s1_y), 2.2, bh, '#666666',
                     'Consensus Samples\n($\\mu$-high / $\\mu$-low)',
                     fontsize=7.5, fill_alpha=0.06, text_color='#333')

    draw_arrow(ax, (5.7, s1_y + bh/2), (6.2, s1_y + bh/2),
               color=C_OURS, linewidth=arr_lw)

    draw_rounded_box(ax, (6.2, s1_y), 2.4, bh, C_OURS,
                     'VLM Dimension\nGenerator',
                     fontsize=8, linewidth=1.0, fill_alpha=0.08)

    draw_arrow(ax, (8.6, s1_y + bh/2), (9.1, s1_y + bh/2),
               color=C_OURS, linewidth=arr_lw)

    # Dimensions output — right-aligned to R_EDGE
    dims_w = R_EDGE - 9.1
    draw_rounded_box(ax, (9.1, s1_y - 0.15), dims_w, bh + 0.3, '#666666',
                     '$\\mathcal{D}_c$: Facade Quality,\n'
                     'Vegetation, Pavement,\n'
                     'Vehicle, Modernity,\n'
                     'Infrastructure, Cleanliness',
                     fontsize=6.5, fill_alpha=0.06, text_color='#333')

    # ════════════════════════════════
    #  STAGE 2
    # ════════════════════════════════
    s2_y = 5.6
    s2_clip_y = 4.3

    draw_rounded_box(ax, (0, 4.7), 2.8, 0.9, C_OURS,
                     'Stage 2\nMulti-Agent Feature\nDistillation',
                     fontsize=8, linewidth=1.2, fill_alpha=0.15,
                     text_color=C_OURS, bold=True)

    draw_rounded_box(ax, (3.5, s2_y), 1.5, bh, '#666666',
                     'Image Pair\n$(x_A, x_B)$',
                     fontsize=7.5, fill_alpha=0.06, text_color='#333')

    agents = [
        ('Observer', '$\\tau$=0.3', 5.4),
        ('Debater', '$\\tau$=0.5', 7.0),
        ('Judge', '$\\tau$=0.1', 8.6),
    ]

    draw_arrow(ax, (5.0, s2_y + bh/2), (5.4, s2_y + bh/2),
               color=C_OURS, linewidth=arr_lw)

    for i, (name, temp, x) in enumerate(agents):
        draw_rounded_box(ax, (x, s2_y), 1.3, bh, C_OURS,
                         f'{name}\n{temp}',
                         fontsize=7.5, linewidth=1.0, fill_alpha=0.12)
        if i < len(agents) - 1:
            next_x = agents[i+1][2]
            draw_arrow(ax, (x + 1.3, s2_y + bh/2), (next_x, s2_y + bh/2),
                       color=C_OURS, linewidth=arr_lw)

    draw_arrow(ax, (9.9, s2_y + bh/2), (10.4, s2_y + bh/2),
               color=C_OURS, linewidth=arr_lw)

    # Semantic scores — right-aligned
    ss_w = R_EDGE - 10.4
    draw_rounded_box(ax, (10.4, s2_y), ss_w, bh, '#666666',
                     '$S(x) \\in [1,10]^7$\nSemantic Scores',
                     fontsize=7, fill_alpha=0.06, text_color='#333')

    # Dashed arrow: Dimensions -> Observer
    dim_arrow_x = 10.6
    dim_arrow_top = s1_y - 0.15
    dim_arrow_bend = s2_y + bh + 0.35
    obs_top_x = 5.8

    ax.plot([dim_arrow_x, dim_arrow_x], [dim_arrow_top, dim_arrow_bend],
            color=C_OURS, linewidth=1.0, linestyle='--', zorder=1)
    ax.plot([dim_arrow_x, obs_top_x], [dim_arrow_bend, dim_arrow_bend],
            color=C_OURS, linewidth=1.0, linestyle='--', zorder=1)
    draw_arrow(ax, (obs_top_x, dim_arrow_bend), (obs_top_x, s2_y + bh),
               color=C_OURS, linewidth=1.0, linestyle='--')
    ax.text(8.2, dim_arrow_bend + 0.15, '$\\mathcal{D}_c$', fontsize=7,
            ha='center', va='center', color=C_OURS, fontstyle='italic')

    # CLIP branch
    draw_rounded_box(ax, (3.5, s2_clip_y), 1.6, bh, C_CLIP,
                     'Frozen CLIP\nViT-L/14',
                     fontsize=7.5, fill_alpha=0.08, text_color=C_CLIP)

    draw_arrow(ax, (4.25, s2_y), (4.25, s2_clip_y + bh),
               color=C_CLIP, linewidth=arr_lw)
    draw_arrow(ax, (5.1, s2_clip_y + bh/2), (5.6, s2_clip_y + bh/2),
               color=C_CLIP, linewidth=arr_lw)

    draw_rounded_box(ax, (5.6, s2_clip_y), 2.2, bh, '#666666',
                     '$\\phi(x) \\in \\mathbb{R}^{768}$\nVisual Embedding',
                     fontsize=7, fill_alpha=0.06, text_color='#333')

    draw_arrow(ax, (7.8, s2_clip_y + bh/2), (8.3, s2_clip_y + bh/2),
               color=C_CLIP, linewidth=arr_lw)

    # Hybrid Vector Fusion — right-aligned
    fusion_x, fusion_y = 8.3, s2_clip_y
    fusion_w = R_EDGE - fusion_x
    draw_rounded_box(ax, (fusion_x, fusion_y), fusion_w, bh, C_OURS,
                     'Hybrid Vector Fusion\n$h = [\\alpha \\cdot \\bar{\\phi},\\; (1{-}\\alpha) \\cdot \\bar{S}]$',
                     fontsize=7, linewidth=1.2, fill_alpha=0.08)

    # Arrow from semantic scores down to fusion
    ss_mid_x = 11.5
    draw_arrow(ax, (ss_mid_x, s2_y), (ss_mid_x, fusion_y + bh + 0.05),
               color=C_OURS, linewidth=arr_lw)

    # ════════════════════════════════
    #  STAGE 3 — compact layout
    # ════════════════════════════════
    s3_y = 1.5
    s3_center = s3_y + bh/2

    draw_rounded_box(ax, (0, 0.95), 2.8, 0.9, C_OURS,
                     'Stage 3\nLocal Manifold\nCalibration (LWRR)',
                     fontsize=8, linewidth=1.2, fill_alpha=0.15,
                     text_color=C_OURS, bold=True)

    # Ref box — taller for multi-line text
    ref_bx, ref_by = 3.3, s3_y + 0.45
    ref_bw, ref_bh = 1.8, 0.85
    ref_box = FancyBboxPatch((ref_bx, ref_by), ref_bw, ref_bh,
                              boxstyle="round,pad=0.05",
                              facecolor=(*matplotlib.colors.to_rgb('#666666'), 0.06),
                              edgecolor='#666666', linewidth=1.0, zorder=2)
    ax.add_patch(ref_box)
    ax.text(ref_bx + ref_bw/2, ref_by + ref_bh/2,
            '$\\mathcal{D}_{ref}$\nHuman Labels\n+ TrueSkill',
            ha='center', va='center', fontsize=6.5, color='#333',
            zorder=3, linespacing=1.3, family='Arial')

    # Pool box — taller, with isolation formula inside
    pool_bx, pool_by = 3.3, s3_y - 0.65
    pool_bw, pool_bh = 1.8, 0.90
    pool_box = FancyBboxPatch((pool_bx, pool_by), pool_bw, pool_bh,
                               boxstyle="round,pad=0.05",
                               facecolor=(*matplotlib.colors.to_rgb('#666666'), 0.06),
                               edgecolor='#666666', linewidth=1.0, zorder=2)
    ax.add_patch(pool_box)
    ax.text(pool_bx + pool_bw/2, pool_by + pool_bh*0.62,
            '$\\mathcal{D}_{pool}$\nTo Calibrate',
            ha='center', va='center', fontsize=6.5, color='#333',
            zorder=3, linespacing=1.3, family='Arial')
    ax.text(pool_bx + pool_bw/2, pool_by + 0.14,
            '$\\mathcal{D}_{ref} \\cap \\mathcal{D}_{pool} = \\varnothing$',
            ha='center', va='center', fontsize=5.5, color=C_FEED,
            fontstyle='italic', zorder=3)

    # Arrows to hybrid diff
    draw_arrow(ax, (ref_bx + ref_bw, ref_by + ref_bh/2),
               (5.6, s3_center + 0.1),
               color='#666', linewidth=arr_lw)
    draw_arrow(ax, (pool_bx + pool_bw, pool_by + pool_bh/2),
               (5.6, s3_center - 0.1),
               color='#666', linewidth=arr_lw)

    # Hybrid Differential Space
    hyb_x = 5.6
    hyb_w = 2.2
    draw_rounded_box(ax, (hyb_x, s3_y - 0.05), hyb_w, bh + 0.1, C_OURS,
                     '$\\Delta_{hybrid}(A,B)$\nDifferential Space',
                     fontsize=7.5, fill_alpha=0.08)

    draw_arrow(ax, (hyb_x + hyb_w, s3_center),
               (hyb_x + hyb_w + 0.4, s3_center),
               color=C_OURS, linewidth=arr_lw)

    # LWRR+KNN merged box — taller for two-line content
    lwrr_x = hyb_x + hyb_w + 0.4
    lwrr_w = 2.8
    lwrr_h = bh + 0.25
    lwrr_y = s3_center - lwrr_h/2
    lwrr_box = FancyBboxPatch((lwrr_x, lwrr_y), lwrr_w, lwrr_h,
                               boxstyle="round,pad=0.05",
                               facecolor=(*matplotlib.colors.to_rgb(C_OURS), 0.08),
                               edgecolor=C_OURS, linewidth=1.0, zorder=2)
    ax.add_patch(lwrr_box)
    ax.text(lwrr_x + lwrr_w/2, lwrr_y + lwrr_h*0.70,
            '$K$-NN ($K$=20) +\nWeighted Ridge Regression',
            ha='center', va='center', fontsize=6.5, color='black',
            zorder=3, family='Arial', linespacing=1.2)
    ax.text(lwrr_x + lwrr_w/2, lwrr_y + lwrr_h*0.22,
            '$\\hat{\\mathbf{w}} = (X^\\top W X + \\lambda I)^{-1} X^\\top W y$',
            ha='center', va='center', fontsize=6.5, color='black', zorder=3)

    # Arrow to output
    out_x = lwrr_x + lwrr_w + 0.4
    out_w = R_EDGE - out_x
    draw_arrow(ax, (lwrr_x + lwrr_w, s3_center),
               (out_x, s3_center),
               color=C_OURS, linewidth=arr_lw)

    # Output box
    draw_rounded_box(ax, (out_x, s3_y), out_w, bh, '#666666',
                     '$\\hat{y}$: Calibrated\nPrediction\n+ $R^2$ Audit',
                     fontsize=6.5, fill_alpha=0.06, text_color='#333')

    # Arrow from fusion down to Stage 3 hybrid diff
    mid_x = fusion_x + fusion_w/2
    hyb_mid = hyb_x + hyb_w/2
    bend_y = s3_y + bh + 1.1
    ax.plot([mid_x, mid_x], [fusion_y, bend_y],
            color=C_OURS, linewidth=arr_lw, zorder=1)
    ax.plot([mid_x, hyb_mid], [bend_y, bend_y],
            color=C_OURS, linewidth=arr_lw, zorder=1)
    draw_arrow(ax, (hyb_mid, bend_y), (hyb_mid, s3_y + bh + 0.05),
               color=C_OURS, linewidth=arr_lw)

    fig.tight_layout(pad=0.2)
    return fig


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(ECCV_DIR, exist_ok=True)

    fig1 = create_figure1()
    path1 = os.path.join(ECCV_DIR, 'fig_comparison.pdf')
    fig1.savefig(path1, bbox_inches='tight', dpi=300, pad_inches=0.05)
    print(f"Saved: {path1}")
    plt.close(fig1)

    fig2 = create_figure2()
    path2 = os.path.join(ECCV_DIR, 'fig_framework.pdf')
    fig2.savefig(path2, bbox_inches='tight', dpi=300, pad_inches=0.05)
    print(f"Saved: {path2}")
    plt.close(fig2)

    print("Done!")
