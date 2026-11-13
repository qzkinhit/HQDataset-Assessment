"""Plot style of the survey, taken from the authors' CARVEPrep and OmniSelect result figures: figures drawn at print width, closed frames, inward ticks, thick lines, hatched bars, horizontal legends above the panels, captions below."""
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
LEGSCALE = 1.15
def setup():
    plt.rcdefaults(); plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .45, "font.family": "DejaVu Sans"})
def box(fig, x, y, w, h, **kw): W_, H_ = fig.get_size_inches(); return fig.add_axes([x / W_, y / H_, w / W_, h / H_], **kw)
def frame(ax, axis="y"):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=TICK, length=2.4, pad=1.8); ax.grid(axis=axis, alpha=.20, lw=.4); ax.set_axisbelow(True)
def key(fig, handles, x, y, cols=2, size=LEGEND):
    W_, H_ = fig.get_size_inches()
    size = size * LEGSCALE     # legends a little larger than the tick labels allow by default
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x / W_, y / H_), ncol=cols, frameon=False, fontsize=size, handlelength=1.0, handletextpad=.27, columnspacing=.5, labelspacing=.12, borderaxespad=0, borderpad=0)
def key_over(fig, ax, handles, cols=None, size=LEGEND, gap=3 / 72, dx=0.0, max_lines=2):
    """A legend centred above one panel, in two lines by default. It takes a third line only where max_lines allows it and two lines are wider than the panel, and otherwise the font shrinks a little until two lines fit."""
    W_, H_ = fig.get_size_inches(); p = ax.get_position(); r = fig.canvas.get_renderer(); n = len(handles); room = p.width * W_ * 1.04 + .25     # a legend may reach over the tick labels on its left
    lines = 1 if n == 1 else 2
    while True:
        c = -(-n // lines); leg = key(fig, handles, (p.x0 + p.x1) / 2 * W_ + dx, p.y1 * H_ + gap, c, size)
        leg._panel = ax
        if leg.get_window_extent(r).width / fig.dpi <= room: return leg
        leg.remove()
        if lines < max_lines: lines += 1
        elif size > LEGEND * .62: size *= .94
        else: leg = key(fig, handles, (p.x0 + p.x1) / 2 * W_ + dx, p.y1 * H_ + gap, c, size); leg._panel = ax; return leg

def key_groups(fig, ax, groups, size=LEGEND, gap=3 / 72, dx=0.0, max_lines=2, sep=.08):
    """Legends of several groups side by side above one panel, each group in its own block of one or two lines and all blocks centred on one height, for example the model families next to the line styles."""
    W_, H_ = fig.get_size_inches(); p = ax.get_position(); r = fig.canvas.get_renderer(); room = p.width * W_ * 1.04 + .25
    lines = [1 if len(g) == 1 else 2 for g in groups]
    while True:
        legs = [fig.legend(handles=g, loc="center left", bbox_to_anchor=(0, 0), ncol=-(-len(g) // l), frameon=False, fontsize=size * LEGSCALE, handlelength=1.0, handletextpad=.27, columnspacing=.5, labelspacing=.12, borderaxespad=0, borderpad=0) for g, l in zip(groups, lines)]
        ex = [L.get_window_extent(r) for L in legs]; ws = [e.width / fig.dpi for e in ex]; hs = [e.height / fig.dpi for e in ex]; tot = sum(ws) + sep * (len(legs) - 1)
        more = [i for i, (g, l) in enumerate(zip(groups, lines)) if l < min(max_lines, len(g))]
        if tot <= room or (not more and size <= LEGEND * .62): break
        for L in legs: L.remove()
        if more: lines[max(more, key=lambda i: ws[i])] += 1
        else: size *= .94
    x = (p.x0 + p.x1) / 2 * W_ + dx - tot / 2; yc = p.y1 * H_ + gap + max(hs) / 2
    for L, w in zip(legs, ws): L.set_bbox_to_anchor((x / W_, yc / H_), transform=fig.transFigure); L._panel = ax; x += w + sep
    return legs
def titles(fig, axes, names):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); H_ = fig.get_figheight(); bottom = min(a.get_tightbbox(r).y0 for a in axes); y = (bottom - 2 * fig.dpi / 72) / (H_ * fig.dpi); out = []
    for a, t in zip(axes, names):
        p = a.get_position(); out.append(fig.text((p.x0 + p.x1) / 2, y, t, ha="center", va="top", fontsize=FS))
    return out
ZH = {"Median": "中位数", "Range": "范围", "Spearman": "Spearman 相关", "Significant": "显著", "Not significant": "不显著", "Joint \u2212 sum (pp)": "联合减单项之和 (百分点)", "Rows": "逐行", "Windows": "窗口", "By run": "按运行", "Macro-F1": "Macro-F1", "Runs per class": "每类独立运行数", "Accuracy": "准确率", "Wrong labels handled": "已处理的错误标签", "Kept share": "保留比例", "Agg.": "聚合", "Rand.": "随机", "Worst": "最差", "Seen": "已见", "Unseen": "未见", "Random": "随机",
 "(a) Repair all errors": "(a) 修复全部错误", "(b) Delete dirty rows": "(b) 删除含错记录", "(c) Static vs. task score": "(c) 静态得分与任务分", "(d) Non-additive repairs": "(d) 修复的不可加性", "(e) Protocol on TEP": "(e) TEP 上的评测协议", "(f) Independent runs": "(f) 独立运行数", "(g) Real label noise": "(g) 真实标签噪声", "(h) Coverage, scale": "(h) 覆盖与规模",
 "Validation records": "验证样本数", "Gain (pp)": "增益 (百分点)", "Cls.": "分类", "Reg.": "回归", "Static": "静态规则", "Year": "年份", "Violations (%)": "违反率 (%)", "Any rule": "任一规则", "Distance": "距离", "Location": "地点", "Test year": "测试年份", "Train year": "训练年份", "Months after training": "训练后月数", "Raw": "原始", "Rules": "规则过滤", "Loss vs. random": "相对随机的损失", "Web": "网页", "Wiki": "维基", "Lowest PPL": "最低困惑度", "Mid PPL": "中段困惑度", "Dedup": "去重", "Both": "两者", "Low PPL": "低困惑度", "Model parameters": "模型参数量", "Wiki loss": "维基损失", "Leaked share": "泄漏比例", "Test loss": "测试损失", "Leaked": "已泄漏", "All": "全部", "Defect rate": "缺陷比例", "Mismatch": "错配", "Missing": "缺失", "Duplicate": "重复", "Coverage": "覆盖", "Scale": "规模",
 "(a) Evidence for a decision": "(a) 证据对决策的价值", "(b) Rule violations": "(b) 规则违反率", "(c) $R^2$ across years": "(c) 跨年 $R^2$", "(d) Time since training": "(d) 训练后的时间", "(e) Filters on C4": "(e) C4 上的过滤", "(f) Filters by model size": "(f) 模型规模与过滤", "(g) Leakage": "(g) 泄漏", "(h) Image-text defects": "(h) 图文缺陷",
 "Static score change": "静态得分变化", "Task change (pp)": "任务分数变化 (百分点)", "Dirty records repaired": "已修复的含错记录", "Dirty records deleted": "已删除的含错记录", "Sorted pairs": "组合, 按变化排序", "Macro-F1 gap (pp)": "Macro-F1 差距 (百分点)", "Window length": "窗口长度", "Macro-F1 gain (pp)": "Macro-F1 增益 (百分点)", "Accuracy gain (pp)": "准确率增益 (百分点)", "Repair": "修复", "Delete": "删除", "Labels": "标注", "(d) Joint minus single gains": "(d) 联合减单项增益", "(e) All data versions": "(e) 全部数据版本", "(f) Repair share": "(f) 修复比例", "(g) Deletion share": "(g) 删除比例", "(h) Protocol vs. data": "(h) 协议与数据", "(i) Split gap by window": "(i) 划分差距与窗口", "(j) Independent runs": "(j) 独立运行数", "(k) Real label noise": "(k) 真实标签噪声", "(l) Coverage, scale": "(l) 覆盖与规模", "Gain over static (pp)": "相对静态规则增益 (百分点)", "Test month": "测试月份", "Train month": "训练月份", "Violation change (pp)": "违反率变化 (百分点)", "R² change": "R² 变化", "Loss gap (×10⁻²)": "损失差 (×10⁻²)", "High PPL": "高困惑度", "Token budget": "词元预算", "Gain over dirty (pp)": "相对脏版本增益 (百分点)", "Synthetic share": "合成比例", "Kept share of the pool": "保留比例", "Recall@5 gain (pp)": "Recall@5 增益 (百分点)", "(a) Choice, classification": "(a) 版本选择, 分类", "(b) Choice, regression": "(b) 版本选择, 回归", "(c) Rule violations": "(c) 规则违反率", "(d) R² over time": "(d) R² 随时间的变化", "(e) Violations and R²": "(e) 违反率与 R²", "(f) Training size, time": "(f) 训练规模与时间", "(g) Filters on C4": "(g) C4 上的筛选", "(h) Filters, model size": "(h) 筛选与模型规模", "(i) Filters, token budget": "(i) 筛选与词元预算", "(j) Real errors, CleanML": "(j) CleanML 真实错误", "(k) Synthetic images": "(k) 合成图像", "(l) Image-text filters": "(l) 图文筛选", "Mean": "平均", "Best": "事后最优", "Passengers": "乘客数", "Rate code": "费率代码", "Duration": "时长", "Fare": "车费", "Speed": "速度", "Above 2 pp": "超过 2 个百分点", "Other months": "其余月份", "Unique": "无近重复", "Task test": "任务测试", "Real": "真实图像", "Top CLIP": "高 CLIP", "Mid CLIP": "中 CLIP", "Low CLIP": "低 CLIP", "Text rule": "文本规则"}
ZH.update({"(a) Choice of version": "(a) 版本选择", "(b) R² over time": "(b) R² 随时间变化", "(c) Violations and R²": "(c) 违反率与 R²", "(d) Training size, time": "(d) 训练规模与时间", "(e) Qwen repair": "(e) Qwen 修复", "(f) Filters, model size": "(f) 筛选与模型规模", "(g) Real errors, CleanML": "(g) CleanML 真实错误", "(g) Synthetic images": "(g) 合成图像", "(h) Image-text filters": "(h) 图文筛选", "(i) Independent runs": "(i) 独立运行数", "(j) Error-rate estimates": "(j) 错误率估计", "Absolute error (pp)": "绝对误差 (百分点)", "Cells per 100 rows": "每 100 行单元格数", "Corrected": "校正", "Damaged": "破坏", "Label errors handled": "已处理的标签错误", "Labeled records": "标注记录数", "Pooled": "直接合并", "Post-strat.": "事后分层", "Qwen high": "Qwen 高分", "Qwen mid": "Qwen 中段", "Repaired": "修复", "Sample": "简单抽样"})
ZH.update({"Gain over static (pp)": "相对静态规则增益 (pp)", "Accuracy gain (pp)": "准确率增益 (pp)", "Recall@5 gain (pp)": "Recall@5 增益 (pp)", "Macro-F1 gain (pp)": "Macro-F1 增益 (pp)", "Macro-F1 gap (pp)": "Macro-F1 差距 (pp)", "Absolute error (pp)": "绝对误差 (pp)", "Gain over dirty (pp)": "相对原表增益 (pp)", "Violation change (pp)": "违反率变化 (pp)", "Real": "真实", "Low PPL": "低困惑度", "Mid PPL": "中困惑度", "High PPL": "高困惑度", "Rules": "规则", "Dedup": "去重", "Both": "两者", "Above 2 pp": "超过 2 pp"})
ZH.update({"(c) Repair vs. delete": "(c) 修复与删除", "Repair path": "修复路径", "Delete path": "删除路径", "(f) Partial repair": "(f) 部分修复", "Share of full gain": "全部修复增益占比", "Static score": "静态得分", "(b) R\u00b2 by train month": "(b) 按训练月份的 R\u00b2", "Mean R\u00b2": "平均 R\u00b2", "Test 2023": "2023 年测试", "Test 2024": "2024 年测试"})
def save_zh(fig, stem):
    """The Chinese version for the report slides, the same figure with translated labels."""
    import matplotlib.text as mtext
    for ax in fig.axes:
        yl = [q.get_text() for q in ax.get_yticklabels()]
        if any(q in ZH for q in yl): ax.set_yticks(ax.get_yticks()); ax.set_yticklabels([ZH.get(q, q) for q in yl])
    for t in fig.findobj(mtext.Text):
        v = t.get_text()
        if v in ZH: t.set_text(ZH[v])
        elif os.environ.get("ZH_MISSING") and any(c.isalpha() for c in v): print("MISS", repr(v))
        t.set_fontfamily(["Hiragino Sans GB", "Arial Unicode MS", "PingFang HK", "DejaVu Sans"])
    return save(fig, stem + "_zh")
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
ZH.update({"Train month (20xx)": "训练月份 (20xx 年)"})
ZH.update({"Train month (yy-mm)": "训练月份 (年-月)"})
ZH.update({"Train month (Jan, Jul)": "训练月份 (1 月和 7 月)"})
ZH.update({"Signif.": "显著", "Not signif.": "不显著"})
ZH.update({"(d) Training size": "(d) 训练规模", "Training records": "训练记录数", "Best per family": "各类模型事后最优", "Errors": "原有错误", "Filled: signif.": "实心为显著"})
ZH.update({"(g) CleanML tables": "(g) CleanML 表"})
ZH.update({"Accuracy vs. real only (pp)": "相对只用真实图像 (pp)", "Loss − random (×10⁻²)": "相对随机的损失差 (×10⁻²)"})
ZH.update({"Loss − random ($\times10^{-2}$)": "相对随机的损失差 ($\times10^{-2}$)"})
ZH.update({"Spearman, score vs. task": "得分与任务性能的 Spearman", "Joint − sum of single (pp)": "联合减单项之和 (pp)", "Signif.": "显著", "Not signif.": "不显著"})
