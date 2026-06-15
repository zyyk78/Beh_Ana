"""
training_convergence.py
========================

独立模块：把多只鼠的同 treatment 数据按"训练次序（ordinal）"对齐，
画"训练次数 → 指标"的收敛折线图。

设计目标：
  1. **不** 修改 BehaviorDataAnalyzer 现有代码 —— 输入只是 `filter_data()` 的
     输出 DataFrame，模块自给自足。
  2. 自动识别每只鼠该 treatment 的"训练段"（连续 T9F 天数 = 一个 block），
     按"同一段内第 N 次"或"全局第 N 次"做对齐。
  3. **每个 metric 输出一张独立 8×5 图**（不再合并到一张 subplot 网格里），
     和 `Documents/VStask/vs_task_learning_curves.py` 一一对应。
  4. **展示用大字风格**（同 `vs_task_switch_latency.png`）：
     - 画布 8×5，白底
     - 标题 20pt bold，轴标签 17pt bold，刻度 13pt
     - 跨鼠均值 ± SE 用 lw=3 彩色实线 + 误差棒
     - 个体鼠淡灰细线
     - 灰色虚线 Convergence ref + italic 灰文字
     - 起 / 终值彩色加粗标注
     - 隐藏 top/right spines，点状淡灰网格
  5. **按指标语义配色**（模拟图 ↔ 真实图一一对应）：
        Reward Ratio / Rate     → 红 (#D55E00)
        Accuracy / Correct Rate → 绿 (#009E73)
        Switching Latency       → 蓝 (#0072B2)
  6. **IQR 离群点去除**：默认对每 (Ordinal) 内做 1.5×IQR；
     可改成整体 / per_mouse / 关掉。

依赖：matplotlib / numpy / pandas（与现有 notebook 相同）。
"""

from __future__ import annotations

from itertools import groupby
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------------
# 0. Nature 期刊风格 rcParams
# ---------------------------------------------------------------------------
def _apply_nature_style():
    """设置 matplotlib rcParams 使输出贴近 Nature 期刊折线图风格。"""
    rc = {
        # 字体（全面放大 + 加粗）
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          12,
        "font.weight":        "bold",
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.labelsize":     13,
        "axes.labelweight":   "bold",
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "legend.title_fontsize": 12,
        # 边线 / spines
        "axes.linewidth":     1.0,
        "axes.edgecolor":     "black",
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        # 网格（淡灰）
        "axes.grid":          True,
        "grid.color":         "#E5E5E5",
        "grid.linewidth":     0.5,
        "grid.linestyle":     "-",
        "axes.axisbelow":     True,   # 网格在数据下方
        # 刻度
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   4.0,
        "ytick.major.size":   4.0,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        # 误差线
        "errorbar.capsize":   3.5,
        # 输出精度
        "pdf.fonttype":       42,   # TrueType (期刊投稿要求)
        "ps.fonttype":        42,
        # 标记
        "lines.markersize":   6,
        "lines.linewidth":    1.8,
    }
    mpl.rcParams.update(rc)


# Nature 风格调色板：Paul-Tol bright 色盲友好色组（Nature 期刊常用，鲜明高级风）
NATURE_PALETTE = {
    "primary":    "#0072B2",   # 鲜亮蓝（主色 — 跨鼠均值线）
    "secondary":  "#D55E00",   # 鲜亮橙红
    "neutral":    "#333333",   # 深灰（接近黑，文字/标签）
    "light":      "#BBBBBB",   # 浅灰（背景鼠细线）
    "accent":     "#E69F00",   # 暖橙
    "teal":       "#009E73",   # 鲜绿
    "purple":     "#CC79A7",   # 粉紫
    "converge":   "#000000",   # 收敛参考线（纯黑）
}

# 当多鼠需要多色时：Paul-Tol bright（鲜明 + 色盲友好）
NATURE_MULTI = [
    "#0072B2",  # 蓝
    "#D55E00",  # 橙红
    "#009E73",  # 绿
    "#E69F00",  # 橙
    "#CC79A7",  # 粉紫
    "#56B4E9",  # 天蓝
    "#F0E442",  # 黄
    "#6A4C93",  # 紫
    "#4D4D4D",  # 灰
    "#7FBC8C",  # 薄荷绿
]


# ---------------------------------------------------------------------------
# 按"指标语义"分配的颜色
# 这套色与 vs_task_learning_curves.py（模拟图）保持完全一致，
# 这样模拟图 ↔ 真实图同一指标颜色互相对应。
#   Reward Ratio / Reward Rate    → 红 (#D55E00)
#   Accuracy / Correct Rate       → 绿 (#009E73)
#   Switching Latency             → 蓝 (#0072B2)
# 未匹配到的 metric 退化到 NATURE_PALETTE["primary"]。
# ---------------------------------------------------------------------------
METRIC_COLOR_MAP = {
    "reward":   "#D55E00",  # 红
    "accuracy": "#009E73",  # 绿
    "correct":  "#009E73",  # 绿（accuracy 同义）
    "switch":   "#0072B2",  # 蓝
    "latency":  "#0072B2",  # 蓝（switch latency 同义）
}


def _metric_color(metric: str) -> str:
    """根据 metric 名称返回与模拟图对齐的颜色。

    匹配规则（大小写不敏感、子串匹配）：
        包含 "reward"  → 红
        包含 "accuracy" / "correct" → 绿
        包含 "switch" / "latency"   → 蓝
        其它 → NATURE_PALETTE["primary"]（深蓝，作为兜底）
    """
    s = str(metric).lower()
    for key, color in METRIC_COLOR_MAP.items():
        if key in s:
            return color
    return NATURE_PALETTE["primary"]


# ---------------------------------------------------------------------------
# 1. 工具：自动找出含有指定 treatment 的所有 mouse_id
# ---------------------------------------------------------------------------
def find_mice_with_treatment(analyzer, treatment: str) -> List[str]:
    """从 BehaviorDataAnalyzer 里找出含指定 treatment 数据的所有鼠。

    用法：
        from training_convergence import find_mice_with_treatment
        mice = find_mice_with_treatment(analyzer, 'T9F')
    """
    mice = sorted(
        analyzer.df[analyzer.df["Treatment"] == treatment]["Mouse_ID"]
        .dropna().unique().tolist()
    )
    return mice


# ---------------------------------------------------------------------------
# 2. 核心：按 ordinal 对齐多鼠数据
# ---------------------------------------------------------------------------
def _assign_ordinals(
    filtered_df: pd.DataFrame,
    treatment: str,
    ordinal_mode: str = "block",  # "block" | "cumulative"
) -> pd.DataFrame:
    """为每行数据打上 ordinal 标签。"""
    if "Mouse_ID" not in filtered_df.columns:
        raise ValueError("filtered_df 必须含 'Mouse_ID' 列")
    if "Treatment" not in filtered_df.columns:
        raise ValueError("filtered_df 必须含 'Treatment' 列")
    if "Date_Parsed" not in filtered_df.columns:
        raise ValueError("filtered_df 必须含 'Date_Parsed' 列")

    df = filtered_df.copy()
    df["Ordinal"] = np.nan
    df["Global_Ord"] = np.nan

    for mid, mdf in df.groupby("Mouse_ID"):
        mdf = mdf.sort_values("Date_Parsed").reset_index()
        treatments = mdf["Treatment"].values

        block_id = 0
        ord_in_block = 0
        global_ord = 0
        prev_t = None
        for i, t in enumerate(treatments):
            if t == treatment:
                if prev_t != t:
                    block_id += 1
                    ord_in_block = 0
                ord_in_block += 1
                global_ord += 1
                if ordinal_mode == "block":
                    df.loc[mdf.iloc[i]["index"], "Ordinal"] = ord_in_block
                else:
                    df.loc[mdf.iloc[i]["index"], "Ordinal"] = global_ord
                df.loc[mdf.iloc[i]["index"], "Global_Ord"] = global_ord
                prev_t = t
            else:
                prev_t = t
    return df


def build_ordinal_table(
    filtered_df: pd.DataFrame,
    treatment: str,
    ordinal_mode: str = "block",
    max_ordinal: Optional[int] = None,
) -> pd.DataFrame:
    """返回只含 target treatment 的、按 ordinal 对齐后的明细表。"""
    df = _assign_ordinals(filtered_df, treatment, ordinal_mode)
    sub = df[df["Treatment"] == treatment].copy()
    sub = sub.dropna(subset=["Ordinal"])
    sub["Ordinal"] = sub["Ordinal"].astype(int)
    if max_ordinal is not None:
        sub = sub[sub["Ordinal"] <= max_ordinal]
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 离群点去除（IQR 方法）
# ---------------------------------------------------------------------------
def remove_outliers_iqr(
    df: pd.DataFrame,
    metrics: Sequence[str],
    method: str = "per_ordinal",   # "per_ordinal" | "overall" | "per_mouse"
    iqr_k: float = 1.5,
) -> pd.DataFrame:
    """IQR 离群点过滤。

    Parameters
    ----------
    df        : 含 Ordinal, Mouse_ID, metrics 的明细表
    metrics   : 要过滤的 metric 列表
    method    :
        - "per_ordinal"  ：在每个 Ordinal 内做 IQR
        - "overall"      ：跨所有 Ordinal 一次性 IQR
        - "per_mouse"    ：每只鼠自己 IQR
    iqr_k     : 1.5（标准 Tukey fence）
    """
    if method is None or method == "none":
        return df
    df = df.copy()
    df["_is_outlier"] = False

    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = df[metric].dropna()
        if vals.empty:
            continue

        if method == "overall":
            groups = [("all", df)]
        elif method == "per_ordinal":
            groups = list(df.groupby("Ordinal"))
        elif method == "per_mouse":
            groups = list(df.groupby("Mouse_ID"))
        else:
            raise ValueError(f"Unknown method: {method}")

        for _, grp in groups:
            v = grp[metric].dropna()
            if len(v) < 4:   # 太少不剔除
                continue
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            mask = (grp[metric] < lo) | (grp[metric] > hi)
            df.loc[mask[mask].index, "_is_outlier"] = True

    return df[~df["_is_outlier"]].drop(columns=["_is_outlier"])


# ---------------------------------------------------------------------------
# 4. 汇总：每个 ordinal 上各鼠的均值/SE
# ---------------------------------------------------------------------------
def summarize_by_ordinal(
    ordinal_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """对每个 (Ordinal, Metric) 计算 Mean / SE / N / Mice。"""
    rows = []
    for metric in metrics:
        valid = ordinal_df[ordinal_df[metric].notna() & (ordinal_df[metric] != 0)]
        for ord_val, grp in valid.groupby("Ordinal"):
            rows.append({
                "Ordinal": int(ord_val),
                "Metric": metric,
                "Mean": grp[metric].mean(),
                "SE":   grp[metric].sem() if len(grp) > 1 else 0.0,
                "N":    len(grp),
                "Mice": ", ".join(sorted(grp["Mouse_ID"].unique())),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. 绘图风格 —— vs_task_switch_latency.png 同款（与 vs_task_learning_curves.py 共用）
# ---------------------------------------------------------------------------
def _apply_show_style() -> None:
    """大字号、加粗、白底，点状淡灰网格——展示用图风格。

    与 `Documents/VStask/vs_task_learning_curves.py::_apply_show_style`
    保持完全一致，这样模拟图和真实图视觉风格完全统一。
    """
    rc = {
        # 字体（大字 + 加粗）
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          14,
        "font.weight":        "bold",
        "axes.titlesize":     20,
        "axes.titleweight":   "bold",
        "axes.labelsize":     17,
        "axes.labelweight":   "bold",
        "xtick.labelsize":    13,
        "ytick.labelsize":    13,
        "legend.fontsize":    14,
        # 边线 / 背景
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "black",
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        # 网格（点状淡灰）
        "axes.grid":          True,
        "grid.color":         "#CCCCCC",
        "grid.linewidth":     0.6,
        "grid.linestyle":     ":",
        "axes.axisbelow":     True,
        # 刻度
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # 误差线
        "errorbar.capsize":   4.0,
        # 输出精度
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        # 默认线宽
        "lines.linewidth":    1.8,
    }
    mpl.rcParams.update(rc)


FIGSIZE = (8, 5)   # 与 vs_task_learning_curves.py 完全一致


def _sanitize_filename(s: str) -> str:
    """把 metric 名转成文件名安全字符串：空格 → 下划线，去掉特殊字符。"""
    keep = "_-."
    return "".join(c if (c.isalnum() or c in keep) else "_" for c in s)


# ---------------------------------------------------------------------------
# 6. 绘图主入口 —— 每个 metric 一张独立 8×5 图，返回 {metric: Figure} 字典
# ---------------------------------------------------------------------------
def plot_training_convergence(
    filtered_df: pd.DataFrame,
    treatment: str,
    metrics: Sequence[str],
    ordinal_mode: str = "block",
    max_ordinal: Optional[int] = None,
    title_template: Optional[str] = None,
    show_individual: bool = True,
    outlier_method: Optional[str] = "per_ordinal",   # None = 不去除
    outlier_iqr_k: float = 1.5,
    min_mice_per_ordinal: int = 1,
    verbose: bool = False,
):
    """把 filter_data() 输出转成"训练收敛曲线"——**每个 metric 一张独立 8×5 图**。

    绘图风格与 `Documents/VStask/vs_task_learning_curves.py` 完全一致
    （即 `vs_task_switch_latency.png` 的展示风格）：
      - 画布 8×5，白底
      - 标题 20pt bold，轴标签 17pt bold，刻度 13pt
      - 跨鼠均值 ± SE 用 lw=3 的彩色实线 + 误差棒
      - 个体鼠淡灰细线（show_individual=True 时）
      - 灰色虚线 Convergence ref + italic 灰文字
      - 起 / 终值彩色加粗标注
      - 隐藏 top/right spines，点状淡灰网格

    Parameters
    ----------
    filtered_df     : analyzer.filter_data(...) 输出
    treatment       : 目标 treatment (e.g. 'T9F')
    metrics         : 要画的指标列表（每个 metric 一张图）
    ordinal_mode    : 'block' (默认) / 'cumulative'
    max_ordinal     : 仅看前 N 次（None = 全部）
    title_template  : 标题模板，可用 `{treatment}` `{metric}` 占位；
                      None 时默认 "{treatment} — {metric} over Training"
    show_individual : 是否画每只鼠的淡色细线
    outlier_method  : None / 'per_ordinal' / 'overall' / 'per_mouse'
    outlier_iqr_k   : 1.5 = 标准 Tukey fence
    min_mice_per_ordinal : 至少几只鼠贡献才画均值（少于则跳过该 ordinal）
    verbose    : True 时打印离群点去除/警告等内部信息；False（默认）静默

    Returns
    -------
    (figs, ordinal_df, summary)
        figs       : Dict[str, matplotlib.figure.Figure] —— 键为 metric 名
        ordinal_df : 离群点去除后的明细表
        summary    : 每 (Ordinal, Metric) 的 Mean/SE/N/Mice
    """
    _apply_show_style()
    ordinal_df = build_ordinal_table(
        filtered_df, treatment,
        ordinal_mode=ordinal_mode,
        max_ordinal=max_ordinal,
    )
    if ordinal_df.empty:
        if verbose:
            print(f"⚠️  没有 Treatment={treatment} 的数据")
        return {}, ordinal_df, pd.DataFrame()

    # 离群点去除
    if outlier_method is not None:
        n_before = len(ordinal_df)
        ordinal_df = remove_outliers_iqr(
            ordinal_df, metrics,
            method=outlier_method, iqr_k=outlier_iqr_k,
        )
        n_removed = n_before - len(ordinal_df)
        if n_removed > 0 and verbose:
            print(f"  [IQR] 去除离群点 {n_removed} 个 "
                  f"({outlier_method}, k={outlier_iqr_k})")

    # 至少 N 只鼠贡献
    if min_mice_per_ordinal > 1:
        mice_per_ord = ordinal_df.groupby("Ordinal")["Mouse_ID"].nunique()
        valid_ords = mice_per_ord[mice_per_ord >= min_mice_per_ordinal].index
        ordinal_df = ordinal_df[ordinal_df["Ordinal"].isin(valid_ords)]

    if ordinal_df.empty:
        if verbose:
            print(f"⚠️  过滤后无数据")
        return {}, ordinal_df, pd.DataFrame()

    summary = summarize_by_ordinal(ordinal_df, metrics)

    if title_template is None:
        title_template = "{treatment} — {metric} over Training"

    # ── 每个 metric 一张图 ──
    figs: dict = {}
    for metric in metrics:
        sub_sum = summary[summary["Metric"] == metric].sort_values("Ordinal")
        if sub_sum.empty:
            if verbose:
                print(f"  ⚠️  metric={metric} 无数据，跳过")
            continue

        # 该 metric 的"语义色"
        m_color = _metric_color(metric)

        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")

        # 1) 个体鼠淡灰细线
        if show_individual:
            for _, (mid, grp) in enumerate(ordinal_df.groupby("Mouse_ID")):
                grp_m = grp.dropna(subset=[metric])
                grp_m = grp_m[grp_m[metric] != 0]
                if grp_m.empty:
                    continue
                grp_m = grp_m.sort_values("Ordinal")
                ax.plot(
                    grp_m["Ordinal"], grp_m[metric],
                    color=NATURE_PALETTE["light"],
                    alpha=0.55, lw=1.2, marker='o', ms=5,
                    markerfacecolor='white', markeredgewidth=0.9,
                    markeredgecolor=NATURE_PALETTE["light"],
                    zorder=3,
                )

        # 2) 跨鼠均值 ± SE（粗实线 + 误差棒，按指标语义色）
        ax.errorbar(
            sub_sum["Ordinal"], sub_sum["Mean"],
            yerr=sub_sum["SE"].fillna(0),
            color=m_color,
            lw=3.0, marker='o', ms=9, markeredgewidth=1.6,
            markerfacecolor='white', markeredgecolor=m_color,
            ecolor=m_color,
            elinewidth=1.6, capsize=4.0, capthick=1.4,
            label='Mean ± SE', zorder=10,
        )

        # 3) 每点鼠数 n=X 标签（小、灰，统一固定像素偏移在点上方 12 px）
        for _, r in sub_sum.iterrows():
            ax.annotate(
                f'n={int(r["N"])}',
                xy=(r["Ordinal"], r["Mean"]),
                xytext=(0, 12), textcoords='offset points',
                fontsize=10, fontweight="bold",
                ha='center', va='bottom', color="#666666",
                zorder=11,
            )

        # 4) 收敛参考线（最后 N 个 ordinal 均值）+ italic 灰文字
        #    文字**统一放在 axes 左上角**（远离曲线 / n=X / 起终值标注）
        if len(sub_sum) >= 2:
            n_tail = min(3, len(sub_sum))
            converge_y = float(sub_sum["Mean"].tail(n_tail).mean())
            ax.axhline(
                converge_y, color="gray", ls="--",
                lw=1.4, alpha=0.7, zorder=5,
            )
            ax.text(
                0.02, 0.97,
                f"Convergence ref: {converge_y:.3f}",
                transform=ax.transAxes,
                color="gray", ha="left", va="top",
                fontsize=13, style="italic", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#BBBBBB",
                          alpha=0.92, lw=0.8),
                zorder=15,
            )
        else:
            converge_y = None

        # 5) 起 / 终值彩色加粗标注（**统一放在数据点下方**，远离上方 n=X 标签）
        if len(sub_sum) >= 2:
            v_start = float(sub_sum.iloc[0]["Mean"])
            v_end   = float(sub_sum.iloc[-1]["Mean"])
            ord_start = int(sub_sum.iloc[0]["Ordinal"])
            ord_end   = int(sub_sum.iloc[-1]["Ordinal"])
            # 自适应数值格式
            fmt = "{:.2f}" if abs(v_end) < 5 else "{:.1f}"
            # 起点放在点的左下方，终点放在点的右下方（远离 n=X 上方标签 + 远离 Convergence ref 左侧文字）
            ax.annotate(
                fmt.format(v_start),
                xy=(ord_start, v_start), xytext=(-4, -18),
                textcoords="offset points",
                fontsize=16, color=m_color, fontweight="bold",
                va="top", ha="right", zorder=20,
            )
            ax.annotate(
                fmt.format(v_end),
                xy=(ord_end, v_end), xytext=(4, -18),
                textcoords="offset points",
                fontsize=16, color=m_color, fontweight="bold",
                va="top", ha="left", zorder=20,
            )

        # ── y 范围：基于真实数据 + 均值±SE，留出顶部 n= 标签空间 ──
        raw_vals = ordinal_df[metric].dropna()
        raw_vals = raw_vals[raw_vals != 0]
        if len(raw_vals) > 0 and sub_sum["Mean"].notna().sum() > 0:
            mean_minus_se = (sub_sum["Mean"] - sub_sum["SE"].fillna(0)).min()
            mean_plus_se  = (sub_sum["Mean"] + sub_sum["SE"].fillna(0)).max()
            ylo = min(raw_vals.min(), mean_minus_se)
            yhi = max(raw_vals.max(), mean_plus_se)
            span = yhi - ylo
            pad = span * 0.18 + 1e-6
            ax.set_ylim(ylo - pad, yhi + pad)

        # ── 美化 ──
        title = title_template.format(treatment=treatment, metric=metric)
        ax.set_xlabel(f"{treatment} training ordinal",
                      fontsize=17, fontweight="bold")
        ax.set_ylabel(metric, fontsize=17, fontweight="bold")
        ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
        ax.set_xticks(sub_sum["Ordinal"].astype(int).unique())
        ax.tick_params(axis='both', which='major', labelsize=13,
                       length=4, width=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(loc="best", fontsize=14, frameon=True)

        fig.tight_layout()
        figs[metric] = fig

    return figs, ordinal_df, summary


# ---------------------------------------------------------------------------
# 7. 一键：在 notebook cell 中调用 —— 多张图依次显示
# ---------------------------------------------------------------------------
def show_in_notebook(
    filtered_df: pd.DataFrame,
    treatment: str,
    metrics: Sequence[str],
    ordinal_mode: str = "block",
    max_ordinal: Optional[int] = None,
    show_individual: bool = True,
    outlier_method: Optional[str] = "per_ordinal",
    outlier_iqr_k: float = 1.5,
    min_mice_per_ordinal: int = 1,
    width: int = 800,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_dpi: int = 300,
    title_template: Optional[str] = None,
    verbose: bool = False,
):
    """在 notebook cell 中依次显示**每个 metric 一张独立 8×5 图**。

    Parameters
    ----------
    save_dir   : 不为 None 时把每张图保存为 PNG + SVG 到该目录
    save_name  : 文件名前缀（不含后缀）；None 时自动用
                 f"training_convergence_{treatment}_{ordinal_mode}"。
                 每张图实际文件名 = f"{save_name}_{metric_safe}.png/.svg"
    save_dpi   : PNG 分辨率（默认 300 dpi）
    verbose    : True 时打印每只鼠的 ordinal 分布、每张图的保存路径与大小；
                 False（默认）只打印一行 header + summary 表格 + 一行保存汇总
    """
    import base64
    import io
    import os
    from IPython.display import HTML, display

    figs, ordinal_df, summary = plot_training_convergence(
        filtered_df, treatment, metrics,
        ordinal_mode=ordinal_mode, max_ordinal=max_ordinal,
        show_individual=show_individual,
        outlier_method=outlier_method,
        outlier_iqr_k=outlier_iqr_k,
        min_mice_per_ordinal=min_mice_per_ordinal,
        title_template=title_template,
        verbose=verbose,
    )
    if not figs:
        return None

    # 打印每只鼠该 treatment 的 ordinal 分布
    cnt = ordinal_df.groupby("Mouse_ID")["Ordinal"].max().astype(int)
    # n_mice = len(cnt)
    # print(f"\n[Training Convergence] treatment = {treatment}, "
    #       f"mode = {ordinal_mode}  ·  {n_mice} mice  ·  {len(figs)} figures")
    # if verbose:
    #     print(f"  各鼠最大 ordinal 分布:")
    #     for mid, n in cnt.items():
    #         print(f"    {mid}: {n} 次")
    # print()
    # print(summary.to_string(index=False))

    # 保存
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f"training_convergence_{treatment}_{ordinal_mode}"
        saved_files = []
        for metric, fig in figs.items():
            metric_safe = _sanitize_filename(metric)
            png_path = os.path.join(save_dir,
                                    f"{save_name}_{metric_safe}.png")
            svg_path = os.path.join(save_dir,
                                    f"{save_name}_{metric_safe}.svg")
            fig.savefig(png_path, format="png", dpi=save_dpi,
                        bbox_inches="tight", facecolor="white")
            fig.savefig(svg_path, format="svg",
                        bbox_inches="tight", facecolor="white")
            saved_files.append((png_path, svg_path))
        # print(f"\n[Save] 目录 = {save_dir}  ·  {len(saved_files)} 指标 × (png+svg)")
        # if verbose:
        #     for png_path, svg_path in saved_files:
        #         print(f"  · {os.path.basename(png_path)}  "
        #               f"({os.path.getsize(png_path)/1024:.1f} KB)")
        #         print(f"  · {os.path.basename(svg_path)}  "
        #               f"({os.path.getsize(svg_path)/1024:.1f} KB)")

    # 嵌入显示（每张图作为独立 <img>）
    def _fig_to_data_uri(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        return ("data:image/png;base64,"
                + base64.b64encode(buf.read()).decode("ascii"))

    html_blocks = []
    for metric, fig in figs.items():
        html_blocks.append(
            f'<img src="{_fig_to_data_uri(fig)}" width="{width}" '
            f'style="display:block;margin:8px 0;"/>'
        )
        plt.close(fig)
    display(HTML("".join(html_blocks)))
    return ordinal_df, summary


if __name__ == "__main__":
    print("training_convergence 模块（展示风格 / 每指标独立图 / IQR 离群点去除）")
    print("  - find_mice_with_treatment(analyzer, 'T9F')")
    print("  - plot_training_convergence(...)  → {metric: Figure}")
    print("  - show_in_notebook(...)")
    print("  - build_ordinal_table(...)")
    print("  - summarize_by_ordinal(...)")
    print("  - remove_outliers_iqr(..., method='per_ordinal')")