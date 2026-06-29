import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Parameters
# -----------------------------
L = 46797
n_fft = 400
hop_length = 160
pad = n_fft // 2  # 200 samples on both sides when center=True

last_frame_position = (L // hop_length) * hop_length  # 46720
last_window_start = last_frame_position - pad         # 46520
last_window_end = last_frame_position + pad - 1       # 46919

# -----------------------------
# Compressed x-coordinate mapping
# -----------------------------
def xmap(x):
    if x <= 700:
        return x + 250  # shift left side so negative padding is visible
    return 1200 + (x - 46000) * 1.8

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)

ax.set_xlim(xmap(-pad) - 120, xmap(L + pad) + 250)
ax.set_ylim(-2.4, 4.4)
ax.axis("off")

timeline_y = 2.6
window_h = 0.9

line_color = "black"
window_edge = "#0B57D0"
window_face = "#EEF5FF"
hop_color = "green"
padding_color = "#2ECC71"
padding_face = "#EAFBF0"

# -----------------------------
# Title
# -----------------------------
ax.text(
    (xmap(-pad) + xmap(L + pad)) / 2,
    4.05,
    "Windowing with center padding",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
)

ax.text(
    (xmap(-pad) + xmap(L + pad)) / 2,
    3.70,
    "Waveform length = 46797, n_fft = 400, hop_length = 160, center = True",
    ha="center",
    va="center",
    fontsize=11,
)

# -----------------------------
# Timeline
# -----------------------------
ax.plot([xmap(-pad), xmap(700)], [timeline_y, timeline_y], color=line_color, lw=2)
ax.plot([xmap(700), xmap(46240) - 180], [timeline_y, timeline_y], color=line_color, lw=2)
ax.plot([xmap(46240) - 120, xmap(L + pad)], [timeline_y, timeline_y], color=line_color, lw=2)

ax.text(
    (xmap(700) + xmap(46240)) / 2,
    timeline_y + 0.18,
    "...",
    ha="center",
    va="bottom",
    fontsize=16,
)

ax.annotate(
    "",
    xy=(xmap(L + pad) + 120, timeline_y),
    xytext=(xmap(L + pad), timeline_y),
    arrowprops=dict(arrowstyle="-|>", lw=2, color=line_color),
)

# -----------------------------
# Padding regions
# -----------------------------
left_padding = patches.Rectangle(
    (xmap(-pad), timeline_y - 0.35),
    xmap(0) - xmap(-pad),
    0.7,
    linewidth=1.4,
    edgecolor=padding_color,
    facecolor=padding_face,
    hatch="////",
)
ax.add_patch(left_padding)

right_padding = patches.Rectangle(
    (xmap(L), timeline_y - 0.35),
    xmap(L + pad) - xmap(L),
    0.7,
    linewidth=1.4,
    edgecolor=padding_color,
    facecolor=padding_face,
    hatch="////",
)
ax.add_patch(right_padding)

ax.text(
    (xmap(-pad) + xmap(0)) / 2,
    timeline_y + 0.9,
    "left padding\n200 samples",
    ha="center",
    va="top",
    fontsize=8,
    color=padding_color,
    fontweight="bold",
)

ax.text(
    (xmap(L) + xmap(L + pad)) / 2,
    timeline_y + 0.9,
    "right padding\n200 samples",
    ha="center",
    va="top",
    fontsize=8,
    color=padding_color,
    fontweight="bold",
)

# -----------------------------
# Ticks
# -----------------------------
ticks = [-200, 0, 160, 320, 480, 46520, 46720, 46797, 46919, 46997]

for t in ticks:
    xt = xmap(t)
    ax.plot([xt, xt], [timeline_y - 0.12, timeline_y + 0.12], color=line_color, lw=1.5)
    ax.text(
        xt,
        timeline_y + 0.22,
        str(t),
        ha="center",
        va="bottom",
        fontsize=8,
    )

# -----------------------------
# Helper function
# -----------------------------
def draw_window(start, y, label):
    end = start + n_fft
    xs = xmap(start)
    xe = xmap(end)

    rect = patches.Rectangle(
        (xs, y),
        xe - xs,
        window_h,
        linewidth=1.6,
        edgecolor=window_edge,
        facecolor=window_face,
    )
    ax.add_patch(rect)

    ax.text(
        (xs + xe) / 2,
        y + window_h / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color=window_edge,
        fontweight="bold",
    )

    ax.plot([xs, xs], [timeline_y, y], color=window_edge, ls="--", lw=1, alpha=0.55)
    ax.plot([xe, xe], [timeline_y, y], color=window_edge, ls="--", lw=1, alpha=0.25)

# -----------------------------
# First windows with center=True
# -----------------------------
draw_window(-200, 1.45, "Window 1\n400 samples")
draw_window(-40, 0.55, "Window 2\n400 samples")

# Hop length
ax.annotate(
    "",
    xy=(xmap(160), 0.25),
    xytext=(xmap(0), 0.25),
    arrowprops=dict(arrowstyle="<->", color=hop_color, lw=1.6),
)

ax.text(
    (xmap(0) + xmap(160)) / 2,
    -0.05,
    "hop_length = 160",
    ha="center",
    va="center",
    fontsize=9,
    color=hop_color,
)

# -----------------------------
# Last centered window
# -----------------------------
draw_window(
    last_window_start,
    0.85,
    "Last window\n293rd frame\n400 samples"
)

# Mark last frame position
ax.plot(
    [xmap(last_frame_position), xmap(last_frame_position)],
    [timeline_y - 0.12, -0.35],
    color="gray",
    ls="--",
    lw=1.4,
)

ax.text(
    xmap(last_frame_position),
    -0.55,
    "last frame position\n46720",
    ha="center",
    va="top",
    fontsize=8,
    color="gray",
)

ax.text(
    xmap(last_window_start),
    0.35,
    "window starts at 46520",
    ha="center",
    va="top",
    fontsize=8,
    color=window_edge,
)

ax.text(
    xmap(last_window_end),
    0.35,
    "window ends at 46919",
    ha="center",
    va="top",
    fontsize=8,
    color=window_edge,
)

ax.text(
    (xmap(L) + xmap(last_window_end)) / 2,
    1.95,
    "uses right padding",
    ha="center",
    va="center",
    fontsize=9,
    color=padding_color,
    fontweight="bold",
)

plt.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.10)
plt.show()