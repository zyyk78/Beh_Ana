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
    """IQR 离群点**标记**（不删除行）。

    返回的 df 带有 ``_is_outlier`` 列（True=离群）。调用方用
    ``df[~df["_is_outlier"]]`` 自行过滤；或保留全部行以便在画个体鼠折线时
    把离群点用虚线 + 红 X 标记出来，保证单鼠数据完整可视化。

    Parameters
    ----------
    df        : 含 Ordinal, Mouse_ID, metrics 的明细表
    metrics   : 要过滤的 metric 列表
    method    :
        - "per_ordinal"  ：在每个 Ordinal 内做 IQR
        - "overall"      ：跨所有 Ordinal 一次性 IQR
        - "per_mouse"    ：每只鼠自己 IQR
    iqr_k     : 1.5（标准 Tukey fence）

    Returns
    -------
    df : 原 df 副本，新增 ``_is_outlier`` bool 列
    """
    df = df.copy()
    df["_is_outlier"] = False
    if method is None or method == "none":
        return df

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

    return df  # 保留 _is_outlier 列；调用方用 ~df["_is_outlier"] 自行过滤


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
    show_outliers: bool = True,
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
    show_outliers   : 是否在个体鼠折线上**可视化**离群点（虚线连接 + 红 X 标记）；
                      True（默认）显示；False 时只画实线穿过正常点，离群段留空。
                      注意：均值的离群点剔除（由 outlier_method 控制）不受此开关影响。
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

    # 离群点标记：保留 _is_outlier 标志，让个体鼠折线能把离群点用虚线+红X画出来
    # 但均值/SE 仍按"干净"数据计算
    if outlier_method is not None:
        n_before = len(ordinal_df)
        ordinal_df_full = remove_outliers_iqr(
            ordinal_df, metrics,
            method=outlier_method, iqr_k=outlier_iqr_k,
        )
        n_removed = int(ordinal_df_full["_is_outlier"].sum())
        if n_removed > 0 and verbose:
            print(f"  [IQR] 标记 {n_removed} 个离群点 "
                  f"({outlier_method}, k={outlier_iqr_k})")
        # 摘要/均值用的"干净"版（剔除 outlier）
        ordinal_df = ordinal_df_full[~ordinal_df_full["_is_outlier"]].drop(columns=["_is_outlier"])
    else:
        ordinal_df_full = ordinal_df.copy()
        ordinal_df_full["_is_outlier"] = False

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

        # 1) 个体鼠淡灰细线 —— 离群点用虚线连接 + 红 X 标记，保证单鼠数据完整
        if show_individual:
            _outlier_label_added = False  # 仅第一个 outlier 散点进图例
            for _, (mid, grp) in enumerate(ordinal_df_full.groupby("Mouse_ID")):
                grp_m = grp.dropna(subset=[metric])
                grp_m = grp_m[grp_m[metric] != 0]
                if grp_m.empty:
                    continue
                grp_m = grp_m.sort_values("Ordinal")

                # 1a) 完整轨迹：虚线穿过**所有**点（含 outlier）—— 仅在 show_outliers=True 时画
                if show_outliers:
                    ax.plot(
                        grp_m["Ordinal"], grp_m[metric],
                        color=NATURE_PALETTE["light"],
                        alpha=0.55, lw=1.2, marker='o', ms=5,
                        markerfacecolor='white', markeredgewidth=0.9,
                        markeredgecolor=NATURE_PALETTE["light"],
                        linestyle='--',   # 虚线：完整轨迹
                        zorder=2,
                    )

                # 1b) 正常段：实线覆盖在虚线之上（被剔除的 outlier 段保留为虚线/留空）
                normal_m = grp_m[~grp_m["_is_outlier"]]
                if not normal_m.empty:
                    ax.plot(
                        normal_m["Ordinal"], normal_m[metric],
                        color=NATURE_PALETTE["light"],
                        alpha=0.75, lw=1.2, marker='o', ms=5,
                        markerfacecolor='white', markeredgewidth=0.9,
                        markeredgecolor=NATURE_PALETTE["light"],
                        linestyle='-',   # 实线：非离群段
                        zorder=3,
                    )

                # 1c) 离群点：红 X 标记（仅第一个进入图例，避免 24 只鼠刷屏）—— 受 show_outliers 控制
                if show_outliers:
                    outliers_m = grp_m[grp_m["_is_outlier"]]
                    if not outliers_m.empty:
                        ax.scatter(
                            outliers_m["Ordinal"], outliers_m[metric],
                            color='#D62728', marker='x', s=80, lw=2.0,
                            alpha=0.9, zorder=4,
                            label='Outlier (IQR)' if not _outlier_label_added else "_nolegend_",
                        )
                        _outlier_label_added = True

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
    show_outliers: bool = True,
    outlier_method: Optional[str] = "per_ordinal",
    outlier_iqr_k: float = 1.5,
    min_mice_per_ordinal: int = 1,
    width: int = 800,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_dpi: int = 300,
    title_template: Optional[str] = None,
    verbose: bool = False,
    print_mice_per_ordinal: bool = True,
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
    print_mice_per_ordinal : True 时按 metric 打印每个 ordinal（iteration）下
                 贡献样本的 mouse_id 列表（即该 iteration 的灰线/均值都用了哪些鼠）。
    """
    import base64
    import io
    import os
    from IPython.display import HTML, display

    figs, ordinal_df, summary = plot_training_convergence(
        filtered_df, treatment, metrics,
        ordinal_mode=ordinal_mode, max_ordinal=max_ordinal,
        show_individual=show_individual,
        show_outliers=show_outliers,
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

    # 按 metric 打印每个 ordinal（iteration）下贡献样本的 mouse_id
    if print_mice_per_ordinal and not summary.empty:
        print(f"\n[Per-Ordinal Mice] treatment = {treatment}, "
              f"mode = {ordinal_mode}  ·  {len(metrics)} metrics")
        for metric in metrics:
            sub = (summary[summary["Metric"] == metric]
                   .sort_values("Ordinal"))
            if sub.empty:
                print(f"\n── {metric} ──  (无数据)")
                continue
            print(f"\n── {metric} ──  ({len(sub)} ordinals)")
            for _, r in sub.iterrows():
                print(f"  Ord {int(r['Ordinal']):>3}  N={int(r['N']):>2}  "
                      f"mice: {r['Mice']}")

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

# ---------------------------------------------------------------------------
#  逐鼠训练轨迹（per-mouse training trajectories）
# ---------------------------------------------------------------------------
def plot_mouse_trajectories(
    filtered_df: pd.DataFrame,
    metrics: Sequence[str],
    ordinal_mode: str = "cumulative",   # 'cumulative'=跨段累加编号; 'block'=同段内重编号; 'session'=全数据集按日期排 session 序
    highlight_outliers: bool = True,
    outlier_iqr_k: float = 3.0,
    width: int = 900,
    height_per_metric: int = 260,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_dpi: int = 300,
    title: Optional[str] = None,
    verbose: bool = False,
):
    """按训练次序对齐、每只鼠一条折线，直观查看单鼠异常波动（不做收敛/均值）。

    Parameters
    ----------
    filtered_df : filter_data(...) 的输出（需含 Mouse_ID / Treatment / Date_Parsed）
    metrics     : 要画的指标列名列表（自动适配 Reward Rate/Accuracy 等别名）
    ordinal_mode:
        'session'    —— 所有行按日期统一编 Session 序（含不同 treatment，适合看整条训练史）
        'cumulative' —— 与收敛图一致：只对 target treatment 累加编号（但此处不过滤 treatment，
                        仍用全部行编号；对齐多 treatment 混合时用这个）
        'block'      —— 同 treatment 连续段内重编号（段间会重置）
    highlight_outliers : 用 IQR(k=outlier_iqr_k) 把单鼠某次的离群点画成红色圆点
    save_dir/save_name : 可选，保存 PNG+SVG

    Returns
    -------
    (fig, tidy_df) : 图 与 整理后的明细表（含 Mouse_ID/Date/Treatment/Ordinal/各 metric）
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    df = filtered_df.copy()
    # 指标列别名适配（与 notebook 口径一致）
    for alias, src in [("Reward Rate", "Reward Ratio"),
                       ("Switch Latency", "Switching Latency")]:
        if alias not in df.columns and src in df.columns:
            df[alias] = df[src]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"metrics 里没有任何一列存在于数据: {metrics}")

    # ---- 编号 ----
    if ordinal_mode == "session":
        df = df.dropna(subset=["Date_Parsed"]).sort_values(["Mouse_ID", "Date_Parsed"]).copy()
        df["Ordinal"] = df.groupby("Mouse_ID").cumcount() + 1
    elif ordinal_mode == "cumulative":
        df = df.dropna(subset=["Date_Parsed"]).sort_values(["Mouse_ID", "Date_Parsed"]).copy()
        df["Ordinal"] = df.groupby("Mouse_ID").cumcount() + 1
    elif ordinal_mode == "block":
        # 真正的 block：按日期排序后，连续相同的 treatment 才是同一段；段内从 1 编号
        df = df.dropna(subset=["Date_Parsed"]).sort_values(["Mouse_ID", "Date_Parsed"]).copy()
        is_new = df["Treatment"].ne(df["Treatment"].groupby(df["Mouse_ID"]).shift())
        df["_blk"] = is_new.groupby(df["Mouse_ID"]).cumsum()
        df["Ordinal"] = df.groupby(["Mouse_ID", "_blk"]).cumcount() + 1
        df = df.drop(columns=["_blk"])
    else:
        raise ValueError("ordinal_mode 只能是 'session' | 'cumulative' | 'block'")

    # 同鼠同天重复行 → 当日均值（一次训练一行）
    agg = df.groupby(["Mouse_ID", "Date_Parsed"], as_index=False).agg(
        {**{m: "mean" for m in metrics}, **{"Ordinal": "first", "Treatment": "first"}}
    )
    agg = agg.sort_values(["Mouse_ID", "Ordinal"])

    # ---- 离群标记（按 metric 的整体 IQR）----
    flags = pd.DataFrame(False, index=agg.index, columns=metrics)
    if highlight_outliers:
        for m in metrics:
            v = agg[m].astype(float)
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - outlier_iqr_k * iqr, q3 + outlier_iqr_k * iqr
            flags[m] = (v < lo) | (v > hi)

    # ---- 画图：每 metric 一行子图，X=Ordinal，每鼠一折线 ----
    mice = sorted(agg["Mouse_ID"].astype(str).unique())
    palette = sns.color_palette("tab10", n_colors=max(len(mice), 1))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(mice)}

    max_ord = int(agg["Ordinal"].max())
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(width/100, height_per_metric*len(metrics)/100),
                             sharex=True, dpi=100)
    if len(metrics) == 1:
        axes = [axes]
    _apply_show_style()

    for ax, m in zip(axes, metrics):
        for mid, g in agg.groupby("Mouse_ID"):
            g = g.sort_values("Ordinal")
            ax.plot(g["Ordinal"], g[m].astype(float), "-o", ms=3.2, lw=1.1,
                    color=color_map[mid], alpha=0.85, label=mid)
            if highlight_outliers:
                out = g[flags.loc[g.index, m]]
                if len(out):
                    ax.scatter(out["Ordinal"], out[m].astype(float), s=26,
                               facecolor="none", edgecolor="red", lw=1.4, zorder=5)
        ax.set_ylabel(m, fontsize=10, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.4)
        # 次序刻度取整数
        ax.set_xticks(range(1, max_ord + 1))
    axes[-1].set_xlabel("Training session #", fontsize=11, fontweight="bold")
    if title:
        axes[0].set_title(title, fontsize=13, fontweight="bold")
    # 图例：一只鼠一个颜色
    handles = [Line2D([0], [0], color=color_map[m], lw=2, marker="o", ms=4, label=m) for m in mice]
    axes[0].legend(handles=handles, ncol=min(len(mice), 12), fontsize=7,
                   loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        name = save_name or "mouse_trajectories"
        png = os.path.join(save_dir, name + ".png")
        svg = os.path.join(save_dir, name + ".svg")
        fig.savefig(png, dpi=save_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
        if verbose:
            print("saved:", png, "|", svg)

    tidy = agg[["Mouse_ID", "Date_Parsed", "Treatment", "Ordinal"] + metrics].copy()
    for m in metrics:
        tidy[m + "_outlier"] = flags[m].values
    if verbose:
        n_out = int(sum(flags[m].sum() for m in metrics))
        print(f"[Mouse Trajectories] {len(mice)} mice x max {max_ord} sessions; {n_out} outliers (red circles)")
    return fig, tidy



# ---------------------------------------------------------------------------
#  Single-session activity reconstruction (per mouse, one day)
# ---------------------------------------------------------------------------
def plot_session_activity(
    pkl_path: str,
    bin_s: float = 60.0,
    show_lick_raster: bool = True,
    lick_raster_max: int = 4000,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_dpi: int = 300,
    verbose: bool = True,
):
    """Reconstruct one session's full activity timeline for a single mouse.

    Layout (top -> bottom):
      1. Reward/omission event rate per bin (SlnL/SlnR up, ROmL/ROmR down)
      2. Mod phases (correct side L/R) as colored background spans
      3. Blackhole intervals as gray spans
      4. Lick raster (optional; left/right ticks) — subsampled if too many
      5. Miniscope frames per bin (recording coverage)

    Returns (fig, summary_df) where summary_df has per-bin counts.
    """
    import pickle
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    FS = 50000.0
    with open(pkl_path, "rb") as f:
        Ev = pickle.load(f)
    if not isinstance(Ev, dict):
        raise ValueError("not a behavior pkl (top-level not dict)")

    all_ts = [t for k, v in Ev.items()
              if isinstance(k, str) and isinstance(v, dict) and v.get("Time_S")
              for t in v["Time_S"]]
    if not all_ts:
        raise ValueError("session has no events")
    t0, t1 = min(all_ts), max(all_ts)
    dur = (t1 - t0) / FS
    nbin = max(int(np.ceil(dur / bin_s)), 1)

    def rel(ts):
        return [(t - t0) / FS for t in (ts or [])]

    def binned(ts):
        b = np.zeros(nbin)
        for t in rel(ts):
            i = min(int(t // bin_s), nbin - 1)
            b[i] += 1
        return b

    g = {k: binned(Ev[k]["Time_S"]) for k in
         ("SlnL", "SlnR", "ROmL", "ROmR", "LickL", "LickR", "MinS")
         if k in Ev}
    for k in ("SlnL", "SlnR", "ROmL", "ROmR", "LickL", "LickR", "MinS"):
        g.setdefault(k, np.zeros(nbin))

    # Mod phases (correct side over time)
    mod_pts = sorted([(t, "L") for t in rel(Ev.get("ModL", {}).get("Time_S") or [])]
                     + [(t, "R") for t in rel(Ev.get("ModR", {}).get("Time_S") or [])])
    phases = []
    if mod_pts:
        # side holds from its Mod event until the next Mod event
        first_side = mod_pts[0][1]
        if mod_pts[0][0] > 0:
            phases.append((0.0, mod_pts[0][0], None))   # pre-first-Mod gap
        for i, (t, side) in enumerate(mod_pts):
            nxt = mod_pts[i + 1][0] if i + 1 < len(mod_pts) else dur
            if nxt > t:
                phases.append((t, nxt, side))
    else:
        phases.append((0.0, dur, None))

    bh = list(zip(rel(Ev.get("Blackhole", {}).get("Time_S") or []),
                  rel(Ev.get("Blackhole", {}).get("Time_E") or [])))

    # ---- figure ----
    n_rows = 4 if show_lick_raster else 3
    heights = [3, 0.6, 0.6, 1.1][:n_rows]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.1 * sum(heights) + 1),
                             sharex=True, gridspec_kw={"height_ratios": heights},
                             dpi=100)

    ax = axes[0]
    x = np.arange(nbin) * bin_s + bin_s / 2
    w = bin_s * 0.9
    ax.bar(x, g["SlnL"], width=w, color="#2E7EB8", label="Reward Left (SlnL)")
    ax.bar(x, g["SlnR"], width=w, color="#7FBFDE", bottom=g["SlnL"], label="Reward Right (SlnR)")
    ax.bar(x, -g["ROmL"], width=w, color="#C44E52", label="Omit Left (ROmL)")
    ax.bar(x, -g["ROmR"], width=w, color="#E8A0A2", bottom=-g["ROmR"], label="Omit Right (ROmR)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Events / bin")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)
    ax.set_title(f"{os.path.basename(pkl_path)}  |  duration {dur/60:.1f} min  |  bin {bin_s:.0f}s",
                 fontsize=12, fontweight="bold")

    # correct-side spans + blackhole spans helper
    def draw_phases(ax, y0=0.0, y1=1.0):
        for a, b, side in phases:
            c = "#BFE3C0" if side == "L" else ("#F6C9C4" if side == "R" else "#E8E8E8")
            ax.axvspan(a, b, ymin=y0, ymax=y1, color=c, alpha=0.5, zorder=0)
    def draw_bh(ax, y0=0.0, y1=1.0):
        for a, b in bh:
            ax.axvspan(a, b, ymin=y0, ymax=y1, color="gray", alpha=0.45, zorder=1)

    draw_phases(ax); draw_bh(ax)

    ax2 = axes[1]
    draw_phases(ax2); draw_bh(ax2)
    for t, side in mod_pts:
        ax2.axvline(t, color="k", lw=1)
    ax2.set_yticks([])
    ax2.set_ylabel("Mod phase")

    ax3 = axes[2]
    draw_phases(ax3); draw_bh(ax3)
    ax3.set_yticks([])
    ax3.set_ylabel("Blackhole")

    if show_lick_raster:
        ax4 = axes[3]
        draw_phases(ax4); draw_bh(ax4)
        for k, row, c in (("LickL", 1, "#2E7EB8"), ("LickR", -1, "#C44E52")):
            ts = rel(Ev[k]["Time_S"])
            if len(ts) > lick_raster_max:
                idx = np.linspace(0, len(ts) - 1, lick_raster_max).astype(int)
                ts = [ts[i] for i in idx]
            ax4.vlines(ts, 0, row * 0.8, color=c, lw=0.4, alpha=0.5)
        ax4.set_ylim(-1, 1)
        ax4.set_yticks([0.4, -0.4], ["Lick L", "Lick R"])
        ax4.set_xlabel("Time in session (s)")

    # Miniscope coverage: put on reward axis as thin line
    if g["MinS"].sum() > 0:
        ax_t = ax.twinx()
        ax_t.plot(x, g["MinS"], color="#555555", lw=0.8, alpha=0.6)
        ax_t.set_ylabel("Miniscope frames/bin", fontsize=8, color="#555555")
        ax_t.tick_params(labelsize=7, colors="#555555")

    sns_despine = True
    for a in axes:
        a.grid(axis="x", ls=":", alpha=0.3)
    fig.tight_layout()

    # ---- per-bin summary ----
    import pandas as pd
    summary = pd.DataFrame({
        "bin_start_s": np.arange(nbin) * bin_s,
        "SlnL": g["SlnL"], "SlnR": g["SlnR"],
        "ROmL": g["ROmL"], "ROmR": g["ROmR"],
        "LickL": g["LickL"], "LickR": g["LickR"],
        "MinS": g["MinS"],
    })

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        name = save_name or (os.path.basename(pkl_path).replace(".pkl", "") + "_activity")
        fig.savefig(os.path.join(save_dir, name + ".png"), dpi=save_dpi,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(save_dir, name + ".svg"), bbox_inches="tight", facecolor="white")
        if verbose:
            print(f"saved: {os.path.join(save_dir, name + '.png')}")

    if verbose:
        n_rew = int(g["SlnL"].sum() + g["SlnR"].sum())
        n_om = int(g["ROmL"].sum() + g["ROmR"].sum())
        print(f"Session: {dur/60:.1f} min | rewards {n_rew} (L {int(g['SlnL'].sum())} / R {int(g['SlnR'].sum())}) "
              f"| omissions {n_om} | blackhole intervals {len(bh)} | mod switches {len(mod_pts)}")
    return fig, summary
