import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Parameters
# -----------------------------
L = 46797
n_fft = 400
hop_length = 160

last_valid_start = ((L - n_fft) // hop_length) * hop_length  # 46240
last_valid_end = last_valid_start + n_fft - 1                # 46639
next_start = last_valid_start + hop_length                   # 46400
ignored_start = last_valid_end + 1                           # 46640

# -----------------------------
# Compressed x-coordinate mapping
# -----------------------------
def xmap(x):
    if x <= 700:
        return x
    return 1200 + (x - 46000) * 1.8

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(15, 7), dpi=200)

ax.set_xlim(-120, xmap(L) + 400)
ax.set_ylim(-2.4, 4.4)
ax.axis("off")

timeline_y = 2.6
window_h = 0.7

line_color = "black"
window_edge = "#0B57D0"
window_face = "#EEF5FF"
hop_color = "green"
ignored_color = "red"

# -----------------------------
# Title
# -----------------------------
ax.text(
    (xmap(0) + xmap(L)) / 2,
    4.05,
    "Windowing problem without center padding",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
)

ax.text(
    (xmap(0) + xmap(L)) / 2,
    3.70,
    "Waveform length = 46797, n_fft = 400, hop_length = 160, center = False",
    ha="center",
    va="center",
    fontsize=11,
)

# -----------------------------
# Timeline
# -----------------------------
ax.plot([xmap(0), xmap(700)], [timeline_y, timeline_y], color=line_color, lw=2)
ax.plot([xmap(700), xmap(46240) - 180], [timeline_y, timeline_y], color=line_color, lw=2)
ax.plot([xmap(46240) - 120, xmap(L)], [timeline_y, timeline_y], color=line_color, lw=2)

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
    xy=(xmap(L) + 120, timeline_y),
    xytext=(xmap(L), timeline_y),
    arrowprops=dict(arrowstyle="-|>", lw=2, color=line_color),
)

# -----------------------------
# Ticks
# -----------------------------
ticks = [0, 160, 320, 480, 46240, 46400, 46640, 46797]

for t in ticks:
    xt = xmap(t)
    ax.plot([xt, xt], [timeline_y - 0.12, timeline_y + 0.12], color=line_color, lw=1.5)
    ax.text(
        xt,
        timeline_y + 0.22,
        str(t),
        ha="center",
        va="bottom",
        fontsize=9,
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
# First two windows
# -----------------------------
draw_window(0, 1.60, "Window 1\n400 samples")
draw_window(160, 0.85, "Window 2\n400 samples")

# Hop length
ax.annotate(
    "",
    xy=(xmap(160), 0.55),
    xytext=(xmap(0), 0.55),
    arrowprops=dict(arrowstyle="<->", color=hop_color, lw=1.6),
)

ax.text(
    (xmap(0) + xmap(160)) / 2,
    0.28,
    "hop_length = 160",
    ha="center",
    va="center",
    fontsize=9,
    color=hop_color,
)

# -----------------------------
# Last meaningful window
# -----------------------------
draw_window(
    last_valid_start,
    0.95,
    "Last meaningful window\n290th window\n400 samples"
)

ax.text(
    xmap(last_valid_start),
    0.25,
    "starts at 46240",
    ha="center",
    va="top",
    fontsize=8,
    color=window_edge,
)

ax.text(
    xmap(last_valid_end),
    0.25,
    "ends at 46639",
    ha="center",
    va="top",
    fontsize=8,
    color=window_edge,
)

# -----------------------------
# Next impossible window start
# -----------------------------
ax.plot(
    [xmap(next_start), xmap(next_start)],
    [timeline_y - 0.12, -0.55],
    color="gray",
    ls="--",
    lw=1.4,
)

ax.text(
    xmap(next_start),
    -0.75,
    "next window\nwould start here\nbut cannot fit",
    ha="center",
    va="top",
    fontsize=8,
    color="gray",
)

# -----------------------------
# Ignored samples
# -----------------------------
ignored_rect = patches.Rectangle(
    (xmap(ignored_start), 0.85),
    xmap(L) - xmap(ignored_start),
    window_h,
    linewidth=1.4,
    edgecolor=ignored_color,
    facecolor="none",
    hatch="////",
)
ax.add_patch(ignored_rect)

ax.text(
    (xmap(ignored_start) + xmap(L)) / 2,
    -0.05,
    "ignored samples\n46640–46797\n157 samples",
    ha="center",
    va="top",
    fontsize=9,
    color=ignored_color,
    fontweight="bold",
)

plt.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.10)
plt.show()