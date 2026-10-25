#!/usr/bin/env python3
"""Render assignment figures exclusively from saved experiment outputs.

Usage: python scripts/make_figures.py --root .
Only the five PDF figures used by the report are written.
No model is trained, selected, or retuned by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "task99-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"tree": "#555555", "bagging": "#0072B2", "boosting": "#D55E00"}
LABELS = {"tree": "Single tree", "bagging": "Bagging", "boosting": "Boosting"}
MODEL_ORDER = ["tree", "bagging", "boosting"]


def as_float(value):
    return np.nan if value is None else float(value)


def depth_label(value):
    return "Unlimited" if value is None else str(value)


def values(records, key):
    return np.asarray([as_float(record.get(key)) for record in records])


def tidy(ax, ylabel=None, xlabel=None, percent=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if percent:
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


class FigureWriter:
    def __init__(self, root):
        self.root = root
        self.figdir = root / "figures"
        self.figdir.mkdir(parents=True, exist_ok=True)
        self.paths = []

    def save(self, fig, stem, caption, notes=None, report=False):
        compact_stems = {"01_baseline_depth", "02_bagging_diagnostics", "03_boosting_history", "04_matched_sizes", "08_depth_bias_variance"}
        if stem in compact_stems:
            fig.set_layout_engine(None)
            fig.set_size_inches(7.2, 2.0)
            for j, ax in enumerate(fig.axes):
                ax.set_position([0.055 + j * 0.245, 0.27, 0.185, 0.47])
                title = ax.get_ylabel().replace("Updated ", "").replace("training ", "")
                ax.set_title("\n".join(textwrap.wrap(title, width=25)), fontsize=8.5, pad=5)
                ax.set_ylabel("")
                xlabel = ax.get_xlabel().replace("Number of trees, B", "Trees (B)").replace("Requested learners", "Learners")
                ax.set_xlabel(xlabel, fontsize=8.5)
                ax.tick_params(axis="both", labelsize=7.5)
                if stem in {"01_baseline_depth", "08_depth_bias_variance"}:
                    ax.tick_params(axis="x", rotation=20)
                    ax.set_xlabel("")
                if stem == "02_bagging_diagnostics":
                    ax.tick_params(axis="x", rotation=45)
                    if j == 1:
                        for annotation in list(ax.texts):
                            annotation.remove()
                if stem == "01_baseline_depth":
                    ax.margins(x=0.1)
                    if j == 0:
                        ax.margins(y=0.16)
                legend = ax.get_legend()
                if legend is not None:
                    if stem in {"01_baseline_depth", "08_depth_bias_variance"}:
                        legend.remove()
                    else:
                        for text in legend.get_texts():
                            text.set_fontsize(6.4)
            if stem == "01_baseline_depth":
                fig.text(0.5, 0.01, "Maximum depth. Minimum leaf size: blue 1 · orange 2 · green 4; selected setting outlined",
                         ha="center", va="bottom", fontsize=7.5)
            if stem == "08_depth_bias_variance":
                fig.text(0.5, 0.01, "Maximum depth. Gray: single tree; blue: 100-tree bagging. Accuracy error bars: one SD.",
                         ha="center", va="bottom", fontsize=7.5)
            self.save(fig, stem + "_compact", caption + " Compact report layout.", notes, report=True)
        else:
            retained = {
                "02_bagging_diagnostics_compact",
                "03_boosting_history_compact",
                "05_bagging_face_cases_compact",
                "06_boosting_high_weight_face",
                "07_masking_divergence_compact",
            }
            if stem in retained:
                directory = self.root / "report" / "figures" if report else self.figdir
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{stem}.pdf"
                fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
                self.paths.append(str(path.relative_to(self.root)))
            plt.close(fig)

    def finish(self):
        print(json.dumps({"figure_count": len(self.paths), "files": self.paths}))


def baseline_figure(summary, writer):
    records = summary["tree_search"]
    depths = sorted({r["depth"] for r in records if r["depth"] is not None})
    if any(r["depth"] is None for r in records):
        depths.append(None)
    leaves = sorted({r["min_samples_leaf"] for r in records})
    palette = ["#0072B2", "#D55E00", "#009E73"]
    x = np.arange(len(depths))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), constrained_layout=True)
    specs = [
        ("validation_accuracy", "Validation accuracy", True),
        ("n_leaves", "Number of leaves", False),
        ("mean_leaf_samples", "Mean training samples / leaf", False),
        ("singleton_fraction", "Leaves with one training sample", True),
    ]
    for ax, (metric, ylabel, percent) in zip(axes.flat, specs):
        for j, leaf in enumerate(leaves):
            selected = {r["depth"]: r for r in records if r["min_samples_leaf"] == leaf}
            y = [as_float(selected[d].get(metric)) for d in depths]
            ax.plot(x, y, "o-", color=palette[j % len(palette)], label=f"Minimum leaf size {leaf}")
        ax.set_xticks(x, [depth_label(d) for d in depths])
        tidy(ax, ylabel, "Maximum depth", percent=percent)
        if metric == "singleton_fraction":
            ax.set_ylim(0, max(0.05, max(as_float(r.get(metric)) for r in records) * 1.1))
    winner = summary["selected"]["tree"]
    d = winner["depth"]
    if d in depths:
        row = next(
            r for r in records
            if r["depth"] == d and r["min_samples_leaf"] == winner["min_samples_leaf"]
        )
        axes[0, 0].scatter([depths.index(d)], [row["validation_accuracy"]], s=100,
                           facecolors="none", edgecolors="black", linewidths=1.1,
                           label="Selected on validation")
    axes[0, 0].legend(fontsize=6.6, loc="best")
    writer.save(fig, "01_baseline_depth",
                "Single-tree search: validation accuracy, leaf count, mean training occupancy per leaf, "
                "and singleton-leaf fraction across depth and minimum leaf size. The outlined point is "
                "the setting selected using validation accuracy. Unlimited denotes max_depth=None.")


def bagging_figure(summary, writer):
    rows = sorted(summary["bagging_search"], key=lambda r: r["B"])
    x = values(rows, "B")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, values(rows, "oob_accuracy"), "o-", color=COLORS["bagging"], label="Out-of-bag")
    ax.plot(x, values(rows, "validation_accuracy"), "s--", color="#009E73", label="Validation")
    tidy(ax, "Accuracy", percent=True)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(x, values(rows, "disagreement"), "o-", color=COLORS["bagging"])
    tidy(ax, "Pairwise tree disagreement", percent=True)
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.text(0.03, 0.06, "Undefined at B = 1", transform=ax.transAxes, fontsize=6.7)
    ax = axes[1, 0]
    ax.plot(x, values(rows, "margin"), "o-", color=COLORS["bagging"])
    tidy(ax, "Mean winning vote margin", percent=True)
    ax = axes[1, 1]
    ax.plot(x, values(rows, "unique_mean"), "o-", color=COLORS["bagging"], label="Observed mean")
    ax.plot(x, values(rows, "unique_theory"), "--", color="#555555", label="Bootstrap expectation")
    tidy(ax, "Distinct training examples / bootstrap")
    ax.legend(fontsize=7)
    selected_b = summary["selected"]["bagging"]["B"]
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(int(b)) for b in x])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Number of trees, B")
        ax.axvline(selected_b, color="black", alpha=0.3, linewidth=0.8, linestyle=":")
    writer.save(fig, "02_bagging_diagnostics",
                "Bagging diagnostics versus ensemble size. Top left: out-of-bag and validation accuracy; "
                "top right: mean pairwise tree disagreement on the clean validation set; bottom left: mean "
                "vote margin on the clean validation set; bottom right: "
                "mean distinct bootstrap examples and their exact theoretical expectation. Disagreement "
                "is undefined for one tree. OOB accuracy uses only training examples with an OOB vote; "
                "coverage is saved for every B. The dotted vertical line marks the validation-selected size.")


def boosting_figure(summary, writer):
    rows = summary["boosting_history"]
    x = values(rows, "round")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), constrained_layout=True)
    specs = [("weighted_error", "Weighted training error", True),
             ("alpha", "SAMME learner coefficient", False),
             ("max_weight", "Largest updated sample weight", True),
             ("top10_weight_sum", "Updated weight of 10 heaviest samples", True)]
    for ax, (key, ylabel, percent) in zip(axes.flat, specs):
        ax.plot(x, values(rows, key), color=COLORS["boosting"], linewidth=1.2)
        tidy(ax, ylabel, "Boosting round", percent=percent)
    writer.save(fig, "03_boosting_history",
                "Diagnostics for the validation-selected SAMME model. Each round shows the weak learner's "
                "weighted error and coefficient, followed by the maximum normalized training sample weight "
                "and the combined normalized weight of the ten heaviest samples after the update.")


def matched_figure(summary, writer):
    records = summary["matched"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), constrained_layout=True)
    specs = [("train_accuracy", "Training accuracy", True),
             ("validation_accuracy", "Validation accuracy", True),
             ("test_accuracy", "Clean test accuracy", True),
             ("fit_seconds", "Training time (seconds)", False)]
    for ax, (key, ylabel, percent) in zip(axes.flat, specs):
        for model in ("bagging", "boosting"):
            rows = sorted([r for r in records if r["model"] == model], key=lambda r: r["n_learners"])
            ax.plot(values(rows, "n_learners"), values(rows, key), "o-", color=COLORS[model], label=LABELS[model])
        sizes = sorted({r["n_learners"] for r in records})
        ax.set_xticks(sizes)
        tidy(ax, ylabel, "Requested learners", percent=percent)
    axes[0, 0].legend(fontsize=7)
    notes = [f"{r['model']} requested {r['n_learners']}, fitted {r['actual_learners']}"
             for r in records if r.get("actual_learners", r["n_learners"]) != r["n_learners"]]
    writer.save(fig, "04_matched_sizes",
                "Bagging and boosting at 25, 50, and 100 requested learners, using each method's selected "
                "base-tree settings. Training, validation, clean test accuracy, and measured training time "
                "are shown. Any early-stopped actual learner counts are recorded in the figure manifest.", notes)


def face_image(images, index, masked=False):
    face = images[int(index)].copy()
    if masked:
        face[24:40, 24:40] = 0
    return face


def show_face(ax, face, title):
    ax.imshow(face, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=7.3, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def top_classes(distributions, required=(), limit=5):
    score = np.max(np.asarray(distributions), axis=0)
    chosen = list(dict.fromkeys(int(i) for i in required))
    for i in np.argsort(-score, kind="stable"):
        if score[i] <= 0:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda i: (-score[i], i))


def face_cases_figure(summary, images, writer):
    available = summary["cases"].get("q1", [])
    chosen = []
    notes = []
    for category in ("correction", "both_wrong"):
        subset = [r for r in available if r["category"] == category]
        chosen.extend(subset[:2])
        if len(subset) < 2:
            notes.append(f"Only {len(subset)} saved {category} case(s); no substitute case invented.")
            chosen.extend([None] * (2 - len(subset)))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.2), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    for j, case in enumerate(chosen):
        if case is None:
            for ax in axes[:, j]:
                ax.axis("off")
            axes[0, j].text(0.5, 0.5, "No qualifying\nsaved example", ha="center", va="center",
                            transform=axes[0, j].transAxes, fontsize=8)
            continue
        masked = case["split"] == "masked"
        category = "Bagging corrects tree" if case["category"] == "correction" else "Both models wrong"
        show_face(axes[0, j], face_image(images, case["index"], masked),
                  f"{category}\n{case['split'].capitalize()} · image {case['index']}")
        axes[0, j].set_xlabel(f"True {case['truth']} · tree {case['tree_prediction']}\nBagging {case['bagging_prediction']}", fontsize=7)
        counts = np.asarray(case["vote_counts"], dtype=float)
        ids = top_classes([counts], [case["truth"], case["tree_prediction"], case["bagging_prediction"]], limit=5)
        labels = [str(i) for i in ids]
        heights = [counts[i] for i in ids]
        other = counts.sum() - sum(heights)
        if other > 0:
            labels.append("Other")
            heights.append(other)
        colors = ["#009E73" if i == case["truth"] else COLORS["bagging"] for i in ids]
        if other > 0:
            colors.append("#bbbbbb")
        ax = axes[1, j]
        bars = ax.bar(np.arange(len(labels)), heights, color=colors, width=0.7)
        ax.set_xticks(np.arange(len(labels)), labels)
        for bar, count in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(round(count))),
                    ha="center", va="bottom", fontsize=6.4)
        ax.margins(y=0.2)
        tidy(ax, "Votes" if j == 0 else None, "Predicted identity")
        ax.set_title(f"Winning vote fraction {counts.max() / counts.sum():.3f}", fontsize=7)
    writer.save(fig, "05_bagging_face_cases",
                "Two examples where bagging corrects the selected single tree and two where both models "
                "are incorrect, when such examples exist. Titles identify clean or masked inputs and global "
                "image indices. Green bars identify votes for the true identity. Leading classes plus the true "
                "and single-tree predicted classes are shown; remaining votes are summed as Other. Full "
                "40-class vote counts are saved in summary.json. Identity labels are zero-based.", notes)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    for ax, case in zip(axes, chosen):
        if case is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "No qualifying\nexample", ha="center", va="center", transform=ax.transAxes)
            continue
        counts = np.asarray(case["vote_counts"], dtype=int)
        ids = np.argsort(-counts, kind="stable")[:3]
        ids = [int(i) for i in ids if counts[i] > 0]
        tally = [f"{i}:{counts[i]}" for i in ids]
        others = int(counts.sum() - counts[ids].sum())
        if others:
            tally.append(f"other:{others}")
        vote_lines = "Votes " + ", ".join(tally[:2])
        if len(tally) > 2:
            vote_lines += "\n" + ", ".join(tally[2:])
        category = "Tree corrected" if case["category"] == "correction" else "Both wrong"
        show_face(ax, face_image(images, case["index"], case["split"] == "masked"),
                  f"{category} · {case['split']} · #{case['index']}")
        ax.set_xlabel(
            f"True {case['truth']}; tree {case['tree_prediction']} ({case.get('tree_confidence', np.nan):.2f})\n"
            f"Bag {case['bagging_prediction']} ({counts.max() / counts.sum():.2f})\n"
            f"{vote_lines}", fontsize=7.5)
    writer.save(fig, "05_bagging_face_cases_compact",
                "Four held-out faces: two corrected by bagging and two missed by both models when available. "
                "Labels give true identity, predictions, winning confidence, and leading vote counts (ID:count); "
                "remaining votes are combined as other. The detailed companion figure gives fuller vote charts.",
                notes, report=True)


def high_weight_figure(summary, predictions, images, writer):
    case = summary["cases"].get("high_weight")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.8, 1.1, 1.1]})
    notes = []
    if not case:
        for ax in axes:
            ax.axis("off")
        axes[1].text(0.5, 0.5, "High-weight example unavailable", ha="center", va="center", transform=axes[1].transAxes)
        notes.append("No high-weight training example saved.")
    else:
        index, position, truth = case["index"], case["train_position"], case["truth"]
        show_face(axes[0], face_image(images, index), f"Training image {index}\nTrue identity {truth}")
        weights = predictions["boost_sample_weights"][:, int(position)]
        x = np.arange(len(weights))
        axes[1].plot(x, weights, color=COLORS["boosting"])
        uniform = 1 / predictions["boost_sample_weights"].shape[1]
        axes[1].axhline(uniform, color="#555555", linestyle="--", linewidth=0.8, label="Initial uniform weight")
        axes[1].set_yscale("log")
        tidy(axes[1], "Normalized sample weight", "Completed updates")
        axes[1].legend(fontsize=6.1, loc="best")
        history = predictions["boost_train_predictions"][:, int(position)]
        rounds = np.arange(1, len(history) + 1)
        correct = history == truth
        axes[2].scatter(rounds[~correct], history[~correct], color=COLORS["boosting"], s=9, label="Incorrect")
        axes[2].scatter(rounds[correct], history[correct], color="#009E73", s=10, label="Correct")
        axes[2].axhline(truth, color="#555555", linestyle=":", linewidth=0.8)
        axes[2].set_ylim(-1, 40)
        tidy(axes[2], "Weak learner predicted identity", "Boosting round")
        axes[2].legend(fontsize=6.1, loc="best")
        axes[0].set_xlabel(f"Incorrect in {np.count_nonzero(~correct)} / {len(history)} rounds", fontsize=7)
    writer.save(fig, "06_boosting_high_weight_face",
                "A training face selected for the largest peak boosting sample weight, its normalized weight "
                "through training, and every weak learner's prediction for that face. Dashed weight line "
                "marks the initial uniform value. Green predictions are correct; orange predictions are "
                "incorrect. The dotted identity line marks its true class.", notes)


def masking_figure(summary, predictions, images, test_indices, y, writer):
    case = summary["cases"].get("q2")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.1]})
    notes = []
    if not case:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 1].text(0.5, 0.5, "No saved masking comparison", ha="center", va="center", transform=axes[0, 1].transAxes)
        notes.append("No masking comparison example saved.")
    else:
        position = int(case["position"])
        index = int(case.get("index", test_indices[position]))
        truth = int(y[index])
        show_face(axes[0, 0], face_image(images, index), f"Clean image {index} · true ID {truth}")
        show_face(axes[0, 1], face_image(images, index, True), "Masked image · central 16 × 16 set to zero")
        axes[0, 2].axis("off")
        lines = ["Prediction and winning confidence", ""]
        for model in MODEL_ORDER:
            clean = predictions[f"{model}_clean"][position]
            masked = predictions[f"{model}_masked"][position]
            lines.extend([LABELS[model], f"  Clean: ID {np.argmax(clean)}, {np.max(clean):.3f}",
                          f"  Masked: ID {np.argmax(masked)}, {np.max(masked):.3f}", ""])
        axes[0, 2].text(0, 1, "\n".join(lines), transform=axes[0, 2].transAxes,
                        fontsize=7.1, va="top", linespacing=1.2)
        for j, model in enumerate(MODEL_ORDER):
            clean = predictions[f"{model}_clean"][position]
            masked = predictions[f"{model}_masked"][position]
            ids = top_classes([clean, masked], [truth, np.argmax(clean), np.argmax(masked)], limit=5)
            x = np.arange(len(ids))
            ax = axes[1, j]
            ax.bar(x - 0.18, clean[ids], 0.36, color=COLORS[model], label="Clean")
            ax.bar(x + 0.18, masked[ids], 0.36, facecolor="white", edgecolor=COLORS[model],
                   hatch="///", linewidth=0.8, label="Masked")
            ax.set_xticks(x, [f"{i}*" if i == truth else str(i) for i in ids])
            ax.set_ylim(0, 1.05)
            ax.set_title(LABELS[model], fontsize=8)
            tidy(ax, "Class vote / probability" if j == 0 else None, "Identity (* = true)", percent=True)
            if j == 0:
                ax.legend(fontsize=6.5)
    writer.save(fig, "07_masking_divergence",
                "One held-out face's clean and centrally masked inputs with the selected models' predicted "
                "identities and winning confidence. Bottom panels show leading class vectors for each model; "
                "filled bars are clean and hatched bars are masked. Confidence is leaf probability for the "
                "single tree, vote fraction for bagging, and normalized weighted vote for boosting. The star "
                "marks the true identity. Only displayed leading classes are plotted.", notes)
    if case:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1, 2.25]})
        show_face(axes[0], face_image(images, index), f"Clean · image {index}")
        show_face(axes[1], face_image(images, index, True), f"Masked · true ID {truth}")
        axes[2].axis("off")
        table_rows = []
        for model in MODEL_ORDER:
            clean = predictions[f"{model}_clean"][position]
            masked = predictions[f"{model}_masked"][position]
            table_rows.append([LABELS[model], f"{np.argmax(clean)} ({clean.max():.3f})",
                               f"{np.argmax(masked)} ({masked.max():.3f})"])
        table = axes[2].table(cellText=table_rows, colLabels=["Model", "Clean ID (q)", "Masked ID (q)"],
                              cellLoc="center", loc="center", colWidths=[0.32, 0.34, 0.34])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.45)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor("#eeeeee")
        axes[2].set_title("Predicted identity and winning confidence q", fontsize=8.5)
        writer.save(fig, "07_masking_divergence_compact",
                    "A clean/masked face comparison with predicted identities and winning confidence for the "
                    "selected models. The detailed companion figure gives their class-vector distributions.", notes,
                    report=True)


def depth_figure(summary, writer):
    records = summary["depth_study"]
    depths = sorted({r["depth"] for r in records if r["depth"] is not None})
    if any(r["depth"] is None for r in records):
        depths.append(None)
    x = np.arange(len(depths))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), constrained_layout=True)
    for ax, key, ylabel, percent in [
        (axes[0, 0], "bias_squared", "Measured squared bias", False),
        (axes[0, 1], "variance", "Measured prediction variance", False),
        (axes[1, 0], "mean_accuracy", "Clean test accuracy across refits", True),
    ]:
        for j, model in enumerate(("tree", "bagging")):
            by_depth = {r["depth"]: r for r in records if r["model"] == model}
            y = np.asarray([as_float(by_depth[d].get(key)) for d in depths])
            yerr = [as_float(by_depth[d].get("std_accuracy")) for d in depths] if key == "mean_accuracy" else None
            ax.bar(x + (j - 0.5) * 0.34, y, 0.34, color=COLORS[model], yerr=yerr,
                   capsize=2, label="Single tree" if model == "tree" else "100-tree bagging",
                   error_kw={"linewidth": 0.7})
        ax.set_xticks(x, [depth_label(d) for d in depths])
        tidy(ax, ylabel, "Maximum depth", percent)
    axes[0, 0].legend(fontsize=6.8)
    tree = {r["depth"]: r for r in records if r["model"] == "tree"}
    bagging = {r["depth"]: r for r in records if r["model"] == "bagging"}
    reduction = [(tree[d]["variance"] - bagging[d]["variance"]) / tree[d]["variance"]
                 if tree[d]["variance"] else np.nan for d in depths]
    axes[1, 1].bar(x, reduction, width=0.6, color=COLORS["bagging"])
    axes[1, 1].axhline(0, color="#555555", linewidth=0.6)
    axes[1, 1].set_xticks(x, [depth_label(d) for d in depths])
    tidy(axes[1, 1], "Fraction of tree variance removed", "Maximum depth", percent=True)
    writer.save(fig, "08_depth_bias_variance",
                "Depth sensitivity under 30 within-identity bootstrap refits: measured squared bias, "
                "prediction-vector variance, mean clean test accuracy, and the fraction of single-tree "
                "variance removed by 100-tree bagging. Accuracy error bars are one sample standard "
                "deviation across the 30 refits, not confidence intervals. No settings are retuned.")


def final_figure(summary, writer):
    table = {r["model"]: r for r in summary["final_table"]}
    repeated = {r["model"]: r for r in summary["biasvariance"]}
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    axes[0].bar(x - 0.18, [table[m]["clean_accuracy"] for m in MODEL_ORDER], 0.36,
                color=[COLORS[m] for m in MODEL_ORDER], label="Clean")
    axes[0].bar(x + 0.18, [table[m]["masked_accuracy"] for m in MODEL_ORDER], 0.36,
                facecolor="white", edgecolor=[COLORS[m] for m in MODEL_ORDER], hatch="///", label="Masked")
    axes[0].set_ylim(0, 1.05)
    tidy(axes[0], "Frozen-model test accuracy", percent=True)
    axes[0].legend(fontsize=6.5)
    b = np.asarray([table[m]["bias_squared"] for m in MODEL_ORDER])
    v = np.asarray([table[m]["variance"] for m in MODEL_ORDER])
    axes[1].bar(x, b, color="#a6cee3", label="Squared bias")
    axes[1].bar(x, v, bottom=b, color="#1f78b4", label="Variance")
    tidy(axes[1], "Prediction-vector squared error")
    axes[1].legend(fontsize=6.5)
    axes[2].errorbar(x, [repeated[m]["mean_accuracy"] for m in MODEL_ORDER],
                     yerr=[repeated[m]["std_accuracy"] for m in MODEL_ORDER],
                     fmt="none", ecolor="#555555", capsize=3, linewidth=1)
    axes[2].scatter(x, [repeated[m]["mean_accuracy"] for m in MODEL_ORDER],
                    c=[COLORS[m] for m in MODEL_ORDER], s=30, zorder=3)
    tidy(axes[2], "Mean clean accuracy ± SD (30 refits)", percent=True)
    for ax in axes:
        ax.set_xticks(x, ["Tree", "Bagging", "Boosting"], rotation=20)
    writer.save(fig, "09_final_comparison",
                "Selected-model comparison: frozen clean and masked test accuracy; squared bias and "
                "prediction variance from the 30-refit experiment; and mean clean test accuracy ± one "
                "sample standard deviation across refits. The stacked middle bars equal measured mean "
                "squared error of the class vectors against one-hot targets, up to numerical precision.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    with np.load(root / "results" / "predictions.npz", allow_pickle=False) as saved:
        predictions = {key: saved[key] for key in saved.files}
    with np.load(root / "data" / "olivetti.npz", allow_pickle=False) as data:
        images, y = data["images"], data["y"]
    with np.load(root / "data" / "split.npz", allow_pickle=False) as split:
        test_indices = split["test"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 7.6, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.markersize": 3.2, "lines.linewidth": 1.1,
    })
    writer = FigureWriter(root)
    bagging_figure(summary, writer)
    boosting_figure(summary, writer)
    face_cases_figure(summary, images, writer)
    high_weight_figure(summary, predictions, images, writer)
    masking_figure(summary, predictions, images, test_indices, y, writer)
    writer.finish()


if __name__ == "__main__":
    main()
