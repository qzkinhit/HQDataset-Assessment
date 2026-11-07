"""Plot style of the result figures, drawn at print width, with closed frames, inward ticks, thick lines, hatched bars, horizontal legends above the panels and captions below."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
W = 176 / 25.4
RED, BLUE, ORANGE, GREEN, GRAY, DARK, PURPLE, TEAL = "#D62728", "#4E79A7", "#F28E2B", "#59A14F", "#888888", "#303030", "#B279A2", "#17A2B8"
FS, TICK, LEGEND = 8.7, 8.0, 7.5
def setup():
    plt.rcdefaults(); plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .45, "font.family": "DejaVu Sans"})
def box(fig, x, y, w, h, **kw): W_, H_ = fig.get_size_inches(); return fig.add_axes([x / W_, y / H_, w / W_, h / H_], **kw)
def frame(ax, axis="y"):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=TICK, length=2.4, pad=1.8); ax.grid(axis=axis, alpha=.20, lw=.4); ax.set_axisbelow(True)
def key(fig, handles, x, y, cols=2, size=LEGEND):
    W_, H_ = fig.get_size_inches()
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x / W_, y / H_), ncol=cols, frameon=False, fontsize=size, handlelength=1.0, handletextpad=.27, columnspacing=.5, labelspacing=.12, borderaxespad=0, borderpad=0)
def key_over(fig, ax, handles, cols=2, size=LEGEND, gap=3 / 72, dx=0.0):
    """A legend centred above one panel."""
    W_, H_ = fig.get_size_inches(); p = ax.get_position(); return key(fig, handles, (p.x0 + p.x1) / 2 * W_ + dx, p.y1 * H_ + gap, cols, size)
def titles(fig, axes, names):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); H_ = fig.get_figheight(); bottom = min(a.get_tightbbox(r).y0 for a in axes); y = (bottom - 2 * fig.dpi / 72) / (H_ * fig.dpi); out = []
    for a, t in zip(axes, names):
        p = a.get_position(); out.append(fig.text((p.x0 + p.x1) / 2, y, t, ha="center", va="top", fontsize=FS))
    return out
def save(fig, stem):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); b = fig.get_tightbbox(r); W_ = fig.get_figwidth()
    if not (b.x0 >= -.05 and b.x1 <= W_ + .05):
        import matplotlib.text as mtext
        for o in list(fig.axes) + list(fig.legends) + list(fig.texts):
            e = o.get_tightbbox(r) if hasattr(o, "get_tightbbox") else o.get_window_extent(r)
            if e is not None and (e.x0 / fig.dpi < -.05 or e.x1 / fig.dpi > W_ + .05): print("overflow", type(o).__name__, [round(e.x0 / fig.dpi, 2), round(e.x1 / fig.dpi, 2)], getattr(o, "get_text", lambda: "")(), [t.get_text() for t in getattr(o, "get_texts", lambda: [])()][:3], o.get_xlabel() if hasattr(o, "get_xlabel") else "")
    assert b.x0 >= -.05 and b.x1 <= W_ + .05, b.bounds
    crop = transforms.Bbox.from_extents(0, b.y0 - .02, W_, b.y1 + .05)
    import matplotlib.text as mtext
    off = set()
    for ax in fig.axes:
        for axis, (u, v) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            off |= {id(l_) for k in axis.get_major_ticks() if not u <= k.get_loc() <= v for l_ in (k.label1, k.label2)}
    for o in fig.findobj(mtext.Text):
        if o.get_text() and o.get_visible() and id(o) not in off:
            e = o.get_window_extent(r)
            if e.width and (e.x0 / fig.dpi < 0 or e.x1 / fig.dpi > W_): print("cut at the page edge:", repr(o.get_text()), [round(e.x0 / fig.dpi, 3), round(e.x1 / fig.dpi, 3)])
    for ext in ("pdf", "png", "svg"): fig.savefig(f"{stem}.{ext}", dpi=450, bbox_inches=crop, pad_inches=0)
    return crop.height
