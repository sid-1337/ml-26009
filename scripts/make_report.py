#!/usr/bin/env python3
"""Write the report's LaTeX source from saved experimental results.

The source uses the supplied official CVPR style and can be compiled to PDF.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path


def number(value, digits=3):
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def percent(value):
    return "N/A" if value is None else f"{100 * float(value):.1f}"


def depth(value):
    return "None" if value is None else str(value)


def label(value):
    return {"tree": "Single tree", "bagging": "Bagging", "boosting": "SAMME"}.get(value, str(value))


def table(headers, rows, caption):
    return {"kind": "table", "headers": headers, "rows": rows, "caption": caption}


def p(text):
    return {"kind": "paragraph", "text": text}


def h(text):
    return {"kind": "heading", "text": text}


def equation(text, latex):
    return {"kind": "equation", "text": text, "latex": latex}


def figure(name, caption, height=175):
    return {"kind": "figure", "name": name, "caption": caption, "height": height}


def build_pages(s):
    """Build the report content from saved experimental results."""
    required = {"metadata", "selected", "tree_search", "bagging_search", "boosting_search",
                "boosting_history", "toy", "matched", "robustness", "cases", "biasvariance",
                "depth_study", "final_table"}
    if required - set(s):
        raise ValueError(f"Missing summary fields: {sorted(required - set(s))}")
    selected = s["selected"]
    tree = selected["tree"]
    bag = selected["bagging"]
    boost = selected["boosting"]
    meta = s["metadata"]
    seed = meta.get("seed", 99)
    seed_status = (f"Seed {seed} is provisional because the complete roll number was not provided."
                   if meta.get("seed_is_provisional", True) else f"Seed {seed} is derived from the provided complete roll number.")
    trees = s["tree_search"]
    bags = s["bagging_search"]
    boosts = s["boosting_search"]
    bv = s["biasvariance"]
    dep = s["depth_study"]
    robust = s["robustness"]
    final = s["final_table"]
    best_tree = next(r for r in trees if r["depth"] == tree["depth"] and r["min_samples_leaf"] == tree["min_samples_leaf"])
    best_bag = next(r for r in bags if r["B"] == bag["B"])
    best_boost = next(r for r in boosts if r["depth"] == boost["depth"] and r["rounds"] == boost["rounds"])
    tree_at_leaf = sorted((r for r in trees if r["min_samples_leaf"] == tree["min_samples_leaf"]),
                          key=lambda r: math.inf if r["depth"] is None else r["depth"])
    shallow_tree, deep_tree = tree_at_leaf[0], tree_at_leaf[-1]
    rb = {r["model"]: r for r in robust}
    toy = s["toy"]
    top_b = max(bags, key=lambda r: r["validation_accuracy"])
    top_oob = max((r for r in bags if r["oob_accuracy"] is not None), key=lambda r: r["oob_accuracy"])
    multi_bags = [r for r in bags if r["B"] > 1]
    top_margin = max(multi_bags, key=lambda r: r["margin"])
    disagreement_range = (min(r["disagreement"] for r in multi_bags), max(r["disagreement"] for r in multi_bags))
    lowest_var = min(bv, key=lambda r: r["variance"])
    highest_acc = max(final, key=lambda r: r["clean_accuracy"])
    most_drop = max(robust, key=lambda r: r["clean_accuracy"] - r["masked_accuracy"])
    stages = s["boosting_history"]
    fin = stages[-1] if stages else {}
    theory = bags[0].get("unique_theory")
    toy_rows = []
    toy_x = toy.get("X", [])
    for i, (truth, pred, before, after) in enumerate(zip(toy["y"], toy["predictions"], toy["weights_before"], toy["weights_after"])):
        xi = toy_x[i] if i < len(toy_x) else []
        feat = ",".join(number(v, 1) for v in xi) if isinstance(xi, list) else number(xi, 1)
        toy_rows.append([str(i), feat, str(truth), str(pred), number(before, 4), number(after, 4)])
    case_counts = {cat: sum(c.get("category") == cat for c in s["cases"].get("q1", []))
                   for cat in ("correction", "both_wrong")}
    full_tree_table = table(
        ["Depth", "Leaf min", "Train %", "Val %", "Test %", "Leaves", "Mean n/leaf", "Singleton %"],
        [[depth(r["depth"]), str(r["min_samples_leaf"]), percent(r["train_accuracy"]), percent(r["validation_accuracy"]),
          percent(r["test_accuracy"]), str(r["n_leaves"]), number(r["mean_leaf_samples"], 2), percent(r["singleton_fraction"])] for r in trees],
        "Table 1. Single-tree grid. Hyperparameters are selected by clean validation accuracy.")
    bag_complete_table = table(["B", "Train %", "Val %", "Test %", "Unique", "OOB %", "Cover %", "Disagree", "Margin"],
        [[str(r["B"]), percent(r["train_accuracy"]), percent(r["validation_accuracy"]), percent(r["test_accuracy"]),
          number(r["unique_mean"], 1), percent(r["oob_accuracy"]), percent(r["oob_coverage"]), number(r["disagreement"]), number(r["margin"])] for r in bags],
        "Table 2. Bagging sweep. OOB uses covered training images; disagreement and margin use clean validation images.")
    boost_table = table(["Depth", "T req/fit", "Train %", "Val %", "Test %"],
        [[str(r["depth"]), f"{r['rounds']}/{r['actual_rounds']}", percent(r["train_accuracy"]), percent(r["validation_accuracy"]), percent(r["test_accuracy"])] for r in boosts],
        "Table 3. SAMME grid, with requested and accepted rounds.")
    matched_table = table(["Model", "Req/fit", "Train %", "Val %", "Test %", "Fit s"],
        [[label(r["model"]), f"{r['n_learners']}/{r['actual_learners']}", percent(r["train_accuracy"]), percent(r["validation_accuracy"]),
          percent(r["test_accuracy"]), number(r["fit_seconds"], 2)] for r in s["matched"]],
        "Table 4. Bagging and SAMME at matched learner counts. Times cover fitting only.")
    robust_table = table(["Model", "Clean %", "Masked %", "Drop pp", "Conf clean", "Conf masked"],
        [[label(r["model"]), percent(r["clean_accuracy"]), percent(r["masked_accuracy"]),
          number(100 * (r["clean_accuracy"] - r["masked_accuracy"]), 1), number(r["clean_confidence"]), number(r["masked_confidence"])] for r in robust],
        "Table 5. Clean and masked versions of the same 80 test faces. Confidence is the mean winning class score.")
    bv_table = table(["Model", "Bias sq.", "Variance", "Acc mean %", "Acc SD pp", "Brier"],
        [[label(r["model"]), number(r["bias_squared"], 4), number(r["variance"], 4), percent(r["mean_accuracy"]),
          number(100 * r["std_accuracy"], 2), number(r["brier_mse"], 4)] for r in bv],
        "Table 6. Thirty paired training resamples with a fixed clean test set. Accuracy SD uses ddof=1.")
    depth_table = table(["Depth", "Model", "Bias sq.", "Variance", "Acc %", "n/leaf", "Disagree"],
        [[depth(r["depth"]), label(r["model"]), number(r["bias_squared"], 4), number(r["variance"], 4),
          percent(r["mean_accuracy"]), number(r.get("mean_leaf_samples"), 2), number(r.get("disagreement"))] for r in dep],
        "Table 7. Depth comparison on 30 shared resamples; each bagging ensemble has 100 trees.")
    final_table = table(["Model", "Clean %", "Masked %", "Bias sq.", "Variance"],
        [[label(r["model"]), percent(r["clean_accuracy"]), percent(r["masked_accuracy"]), number(r["bias_squared"], 4), number(r["variance"], 4)] for r in final],
        "Table 8. Selected models: original clean/masked accuracy and repeated-refit bias and variance.")

    pages = []
    pages.append({"title": "Experimental setup", "full": [], "left": [
        h("Abstract"),
        p(f"This study compares a decision tree, bagging, and multiclass SAMME on raw Olivetti face pixels. The split has 240 training, 80 validation, and 80 test images from 40 identities. Validation selects tree depth {depth(tree['depth'])}, minimum leaf {tree['min_samples_leaf']}, {bag['B']} bagged trees, and SAMME depth {boost['depth']} with {boost['rounds']} rounds. Clean-test accuracies are {percent(rb['tree']['clean_accuracy'])}%, {percent(rb['bagging']['clean_accuracy'])}%, and {percent(rb['boosting']['clean_accuracy'])}%. A central mask tests occlusion; 30 paired training resamples measure bias and variance."),
        h("1. Data and split"),
        p("Olivetti has 400 grayscale images, 10 per identity, at 64 x 64 resolution [1]. numpy.random.default_rng permutes each identity's images into six training, two validation, and two test examples. The same indices are reused throughout. Inputs are 4,096 raw pixels, without PCA or feature learning."),
        p("The mask zeros rows 24:40 and columns 24:40 of the test images before flattening. It removes 256 pixels, or 6.25%, from each of the same 80 faces. Training and validation images stay clean."),
        h("2. Model selection"),
        p("Only clean validation accuracy selects hyperparameters. Ties prefer shallower depth, larger minimum leaf size, then fewer learners or rounds; None follows all finite depths. Test scores play no part in selection. Class vectors use labels 0-39 in order, with the first class winning score ties.")], "right": [
        h("Run status"),
        p(f"{seed_status} Code and report were generated with Codex."),
        h("3. Methods"),
        p("DecisionTreeClassifier supplies individual trees [2]. The ensemble code implements bootstrap draws, voting, out-of-bag (OOB) predictions, and SAMME weight updates directly; it does not use library ensemble classifiers."),
        p("Bagging fits each tree to 240 training positions drawn with replacement and averages hard votes [3]. SAMME fits trees under changing sample weights and combines their hard predictions using positive learner coefficients [4]. Both return normalized class vectors."),
        h("4. Experimental controls"),
        p("The bagging sweep uses prefixes of one 200-tree ensemble. Timing comparisons fit each size afresh. All methods share the same 30 within-identity training resamples and fixed test set, without retuning. Keeping model RNG streams fixed across repetitions isolates the effect of changing training images."),
        p("The resampling study concerns this training pool and split, not new identities or test sets. Robustness covers one central zero mask. Winning vote scores are treated as confidence measures, without assuming calibration.")]})

    pages.append({"title": "Single tree and bootstrap aggregation", "full": [
        figure("02_bagging_diagnostics", "Figure 1. OOB and validation accuracy, tree disagreement, vote margin, and distinct bootstrap examples versus B.", 150),
        full_tree_table], "left": [
        h("5. Baseline selection"),
        p(f"The validation winner has depth {depth(tree['depth'])} and minimum leaf {tree['min_samples_leaf']}, with train/validation/test accuracy {percent(best_tree['train_accuracy'])}/{percent(best_tree['validation_accuracy'])}/{percent(best_tree['test_accuracy'])}%. Its {best_tree['n_leaves']} leaves average {number(best_tree['mean_leaf_samples'], 2)} training samples; {percent(best_tree['singleton_fraction'])}% contain only one sample."),
        p(f"With the leaf minimum held at {tree['min_samples_leaf']}, increasing depth from {depth(shallow_tree['depth'])} to {depth(deep_tree['depth'])} raises training accuracy from {percent(shallow_tree['train_accuracy'])}% to {percent(deep_tree['train_accuracy'])}%. Leaves increase from {shallow_tree['n_leaves']} to {deep_tree['n_leaves']}, while average occupancy falls from {number(shallow_tree['mean_leaf_samples'], 2)} to {number(deep_tree['mean_leaf_samples'], 2)}. The shallow tree underfits: its few large leaves cannot separate many identities.")], "right": [
        h("6. Bootstrap coverage"),
        p(f"A training example is absent from n=240 replacement draws with probability (1-1/n)^n. Hence E[distinct] = n[1-(1-1/n)^n] = {number(theory, 3)}. Table 2's averages stay close to this expectation. Repeated draws count as repeated fitting samples."),
        p(f"Validation chooses B={bag['B']} at {percent(best_bag['validation_accuracy'])}%; clean-test accuracy reaches {percent(best_bag['test_accuracy'])}%. OOB accuracy, disagreement, and margin help interpret this choice but do not select the size.")]})

    corrections = [c for c in s["cases"].get("q1", []) if c["category"] == "correction"]
    shared_errors = [c for c in s["cases"].get("q1", []) if c["category"] == "both_wrong"]
    correction_text = (f"Figure 2 contains {case_counts['correction']} bagging corrections and {case_counts['both_wrong']} shared errors. ")
    if corrections:
        correction_text += " ".join(
            f"For image {c['index']}, the tree predicts {c['tree_prediction']}, while the true class {c['truth']} wins {int(c['vote_counts'][c['truth']])}/{bag['B']} bagging votes."
            for c in corrections)
        correction_text += (" A correct plurality can recover an error without a majority."
                            if any(c["vote_counts"][c["truth"]] <= bag["B"] / 2 for c in corrections)
                            else " Combining the trees overturns these single-tree errors.")
    shared_error_text = " ".join(
        f"Image {c['index']} belongs to class {c['truth']}, but class {c['bagging_prediction']} leads with {int(c['vote_counts'][c['bagging_prediction']])}/{bag['B']} votes."
        for c in shared_errors)
    shared_error_text += (" These dispersed votes still produce the wrong winner. Aggregation helps when errors differ across trees, "
                          "but cannot remove every shared confusion. Image conditions and complete class counts accompany the examples.")

    pages.append({"title": "Bagging diagnostics and error cases", "full": [
        bag_complete_table,
        figure("05_bagging_face_cases", "Figure 2. Two bagging corrections and two shared errors, with labels, confidence, and leading vote counts.", 150)], "left": [
        h("7. Reading the diagnostics"),
        p(f"At B={bag['B']}, OOB accuracy is {percent(best_bag['oob_accuracy'])}% with {percent(best_bag['oob_coverage'])}% coverage. Validation disagreement is {number(best_bag['disagreement'])} and mean margin is {number(best_bag['margin'])}. Validation first reaches its maximum at B={top_b['B']}."),
        p(f"OOB also peaks at B={top_oob['B']}, {'supporting' if top_oob['B'] == bag['B'] else 'differing from'} the validation choice. For B>1, disagreement varies only from {number(disagreement_range[0])} to {number(disagreement_range[1])}; the largest margin occurs at B={top_margin['B']}. Diversity alone does not track the accuracy gain. B=1 even has margin one despite low accuracy, because its single vote is unanimous."),
        p("Disagreement averages differing predictions across unordered tree pairs on validation images; it is undefined at B=1. Margin is the average winning minus runner-up vote fraction, including classes with zero votes. OOB accuracy excludes images with no eligible votes.")], "right": [
        h("8. What the error cases show"),
        p(correction_text),
        p(shared_error_text)]})

    pages.append({"title": "SAMME implementation and candidate search", "full": [
        figure("03_boosting_history", "Figure 3. Selected SAMME weighted error, learner coefficient, maximum weight, and top-10 weight mass by round.", 150)], "left": [
        h("9. Update equations"),
        equation("Initially w_i=1/n. For a weak prediction h_t, epsilon_t=sum_i w_i 1[h_t(x_i) != y_i]. The coefficient is alpha_t=log((1-epsilon_t)/epsilon_t)+log(K-1). Update w_i <- w_i exp(alpha_t 1[h_t(x_i) != y_i]), then normalize. Score S_k(x)=sum_t alpha_t 1[h_t(x)=k]; p_k=S_k/sum_t alpha_t.",
                 r"""\begin{align*}
\epsilon_t &= \sum_i w_i^{(t)}\mathbf{1}[h_t(x_i)\ne y_i],\\
\alpha_t &= \log\frac{1-\epsilon_t}{\epsilon_t}+\log(K-1),\\
w_i^{(t+1)} &= \frac{w_i^{(t)}e^{\alpha_t\mathbf{1}[h_t(x_i)\ne y_i]}}{\sum_j w_j^{(t)}e^{\alpha_t\mathbf{1}[h_t(x_j)\ne y_j]}},\\
p_k(x)&=\frac{\sum_t\alpha_t\mathbf{1}[h_t(x)=k]}{\sum_t\alpha_t}.
\end{align*}"""),
        p("Weights start at 1/n. A learner with error at least 1-1/K stops the run; rejection in the first round invalidates the candidate. A perfect learner uses machine epsilon to keep its coefficient finite, then stops. Tables distinguish requested and accepted rounds."),
        h("10. One toy round"),
        p(f"The {len(toy['y'])}-sample, three-class toy uses the same update code as the face experiment. Its weighted error is {number(toy['weighted_error'], 6)} and alpha is {number(toy['alpha'], 6)}. The table shows weights before and after normalization."),
        table(["i", "x", "y", "h", "w before", "w after"], toy_rows,
              "Toy trace. A wrong prediction is multiplied by exp(alpha) before normalization; all weights sum to one.")], "right": [
        boost_table,
        h("11. Selected model and weight evolution"),
        p(f"Validation selects depth {boost['depth']} and {boost['rounds']} rounds, all {boost['actual_rounds']} accepted. The grid shows that more rounds do not consistently improve held-out accuracy."),
        p(f"In the final round, error is {number(fin.get('weighted_error'), 4)} and alpha is {number(fin.get('alpha'), 4)}. The largest weight reaches {number(fin.get('max_weight'), 4)}; the ten heaviest samples hold {percent(fin.get('top10_weight_sum'))}% of the mass. Errors use entering weights, while weight summaries follow the update. The top-ten membership can change each round.")]})

    high = s["cases"].get("high_weight", {})
    high_history = high.get("history", [])
    high_mistakes = sum(bool(r.get("misclassified")) for r in high_history)
    matched100 = {r["model"]: r for r in s["matched"] if r["n_learners"] == 100}
    bag_wins = sum(next(r for r in s["matched"] if r["model"] == "bagging" and r["n_learners"] == size)["test_accuracy"] >
                   next(r for r in s["matched"] if r["model"] == "boosting" and r["n_learners"] == size)["test_accuracy"]
                   for size in (25, 50, 100))
    pages.append({"title": "Bagging versus SAMME", "full": [
        matched_table,
        figure("06_boosting_high_weight_face", "Figure 4. A high-weight training face, its weight history, and the weak learners' predictions.", 165)], "left": [
        h("12. Accuracy and training time"),
        p("Table 4 compares fresh fits at 25, 50, and 100 learners, using the selected tree settings and weak depth. Each timing is one wall-clock measurement on the same machine, excluding data loading, prediction, and plotting."),
        p(f"Bagging is more accurate at {bag_wins} of the three sizes, but its 100-tree fit takes {number(matched100['bagging']['fit_seconds']/matched100['boosting']['fit_seconds'], 2)} times as long as SAMME. Matching learner counts does not match capacity: bagging uses depth {depth(tree['depth'])}; SAMME uses depth {boost['depth']}, allowing at most {2**boost['depth']} leaves for 40 identities per weak learner. Boosting is sequential, whereas bagging's trees can be trained independently.")], "right": [
        h("13. Why a face receives large weight"),
        p(f"Image {high.get('index', 'not recorded')} belongs to class {high.get('truth', 'not recorded')} and occupies training position {high.get('train_position', 'not recorded')}. Weak learners misclassify it in {high_mistakes}/{len(high_history)} rounds. Its peak weight, {number(high.get('peak_weight'), 5)}, is {number(high.get('peak_weight', 0)*240, 1)} times the initial 1/240. It is selected by the highest weight reached in any round."),
        p("Each mistake multiplies its weight by exp(alpha) before normalization. Correct rounds receive no multiplier and its normalized weight falls, producing the sharp drops in Figure 4. Repeated mistakes bring it back into focus. This explains the learner's attention to the image, without assuming its label is wrong.")]})

    q2 = s["cases"].get("q2", {})
    improved_under_mask = [r for r in robust if r["masked_accuracy"] > r["clean_accuracy"]]
    improvement_text = " ".join(f"{label(r['model'])} improves by {number(100*(r['masked_accuracy']-r['clean_accuracy']), 2)} pp under this mask."
                                for r in improved_under_mask)
    q2_text = (f"Image {q2.get('index', 'not recorded')} has true class {q2.get('truth', 'not recorded')}. "
               f"Bagging predicts {q2.get('bagging_clean_prediction', 'N/A')} before masking and {q2.get('bagging_masked_prediction', 'N/A')} after; "
               f"SAMME changes from {q2.get('boosting_clean_prediction', 'N/A')} to {q2.get('boosting_masked_prediction', 'N/A')}. ")
    if "bagging_tree_change_fraction" in q2:
        q2_text += (f"The mask changes {percent(q2['bagging_tree_change_fraction'])}% of bagging predictions and "
                    f"{percent(q2['boosting_alpha_weighted_tree_change_fraction'])}% of SAMME's coefficient mass. "
                    f"Bagging disagreement changes from {number(q2.get('bagging_clean_pairwise_disagreement'))} "
                    f"to {number(q2.get('bagging_masked_pairwise_disagreement'))}.")
    else:
        q2_text += "The case record includes vote counts and normalized weighted scores."
    pages.append({"title": "Central-mask robustness and disagreement", "full": [
        robust_table,
        figure("07_masking_divergence", "Figure 5. The same held-out face before and after masking; predictions and normalized winning scores are shown.", 155)], "left": [
        h("14. Accuracy under occlusion"),
        p("Table 5 uses the same clean-trained models on both test conditions, without adaptation to masked images. Confidence averages winning scores over all 80 faces, including mistakes. Because the methods define scores differently, equal confidence need not imply equal calibration."),
        p(f"The tree's masked mean confidence is {number(rb['tree']['masked_confidence'])}, despite accuracy of {percent(rb['tree']['masked_accuracy'])}%. Routing an occluded face to a pure leaf preserves confidence even when the identity is wrong."),
        p(f"{label(most_drop['model'])} suffers the largest drop, {number(100*(most_drop['clean_accuracy']-most_drop['masked_accuracy']), 1)} percentage points. {improvement_text} These effects concern just 80 faces and one mask; any improvement here does not establish a general benefit from occlusion.")], "right": [
        h("15. Different responses to a missing region"),
        p(q2_text),
        p("Bagging's equally weighted votes retain the correct lead here, despite substantial disagreement. SAMME's reweighted training and coefficient-weighted votes produce a different winner. The vote changes explain this example's response; a single face cannot establish that training-weight concentration causes the robustness difference.")]})

    depth_groups = {}
    for r in dep:
        depth_groups.setdefault(str(r["depth"]), {})[r["model"]] = r
    comparison_text = []
    improvements = []
    for d, records in depth_groups.items():
        if "tree" in records and "bagging" in records:
            tr, ba = records["tree"], records["bagging"]
            comparison_text.append(f"Depth {depth(None if d == 'None' else d)}: accuracy change {number(100*(ba['mean_accuracy']-tr['mean_accuracy']), 2)} pp; variance reduction {number(tr['variance']-ba['variance'], 4)}.")
            improvements.append((None if d == "None" else int(d), ba["mean_accuracy"] - tr["mean_accuracy"]))
    if improvements:
        deepest = max(improvements, key=lambda x: math.inf if x[0] is None else x[0])
        max_gain = max(g for _, g in improvements)
        deepest_conclusion = (f"The deepest condition ({depth(deepest[0])}) "
            + ("gives" if math.isclose(deepest[1], max_gain, abs_tol=1e-12) else "does not give")
            + f" the largest accuracy gain ({number(100 * max_gain, 2)} pp).")
    else:
        deepest_conclusion = "No paired depth records were available."
    ordered_depths = sorted(depth_groups, key=lambda d: math.inf if d == "None" else int(d))
    shallow_condition = depth_groups[ordered_depths[0]]
    deep_condition = depth_groups[ordered_depths[-1]]
    st, sb = shallow_condition["tree"], shallow_condition["bagging"]
    dt, db = deep_condition["tree"], deep_condition["bagging"]
    depth_evidence = (
        f"At depth {ordered_depths[0]}, bagging changes squared bias from {number(st['bias_squared'], 4)} to {number(sb['bias_squared'], 4)} "
        f"and mean leaf occupancy from {number(st['mean_leaf_samples'], 2)} to {number(sb['mean_leaf_samples'], 2)}; disagreement is {number(sb['disagreement'])}. "
        f"At depth {ordered_depths[-1]}, bias changes from {number(dt['bias_squared'], 4)} to {number(db['bias_squared'], 4)} "
        f"while variance falls from {number(dt['variance'], 4)} to {number(db['variance'], 4)}; "
        f"occupancy is {number(dt['mean_leaf_samples'], 2)} versus {number(db['mean_leaf_samples'], 2)} and disagreement is {number(db['disagreement'])}. "
        "Here, the large variance reduction comes with higher squared bias.")
    bag_depths = [r for r in dep if r["model"] == "bagging"]
    min_var_depth = min(bag_depths, key=lambda r: r["variance"])
    max_acc_depth = max(bag_depths, key=lambda r: r["mean_accuracy"])
    if min_var_depth["mean_accuracy"] < max_acc_depth["mean_accuracy"]:
        variance_counterexample = (
            f"Bagging at depth {depth(min_var_depth['depth'])} has variance {number(min_var_depth['variance'], 4)} and mean accuracy {percent(min_var_depth['mean_accuracy'])}%, "
            f"whereas depth {depth(max_acc_depth['depth'])} has larger variance {number(max_acc_depth['variance'], 4)} but mean accuracy {percent(max_acc_depth['mean_accuracy'])}%: "
            "lower variance does not guarantee better accuracy. High disagreement alone is also insufficient.")
    else:
        variance_counterexample = ("These variance and accuracy rankings concern this model set and data; "
                                  "the squared-vector decomposition does not determine the accuracy ranking.")
    pages.append({"title": "Bias and variance across 30 refits", "full": [
        bv_table,
        depth_table], "left": [
        h("16. Resampling procedure"),
        p("Each of 30 repetitions draws six images with replacement from each identity's original six training images. Every model and depth uses that repetition's 240 training positions, with no retuning. All refits predict the same 80 clean test faces."),
        equation("Let p_ri be the 40-class prediction vector for repetition r and test image i, and mean_p_i its average over 30 fits. Bias squared is (1/N) sum_i ||mean_p_i-e_i||^2. Variance is (1/(N R)) sum_ri ||p_ri-mean_p_i||^2. e_i is the one-hot target, N=80 and R=30.",
                 r"""For the one-hot target $e_i$, $N=80$, and $R=30$,
\begin{align*}
\bar p_i &= \frac{1}{R}\sum_r p_{ri},\\
\mathrm{Bias}^2 &= \frac1N\sum_i\|\bar p_i-e_i\|_2^2,\\
\mathrm{Variance} &= \frac1{NR}\sum_{r,i}\|p_{ri}-\bar p_i\|_2^2.
\end{align*}"""),
        p("Mean squared vector error equals squared bias plus variance, within floating-point tolerance. Classification accuracy has no such decomposition.")], "right": [
        h("17. Depth effects and bagging benefit"),
        p(" ".join(comparison_text)),
        p(deepest_conclusion),
        p(depth_evidence),
        p("Tree vectors use predict_proba; bagging uses vote fractions; SAMME uses normalized coefficient-weighted votes. All follow the same class order. Accuracy SD uses ddof=1 across refits, while vector variance uses denominator R.")]})

    conclusion = (f"Among the three selected models, {label(lowest_var['model'])} has the lowest variance ({number(lowest_var['variance'], 4)}) and squared bias {number(lowest_var['bias_squared'], 4)}. "
                  f"{label(highest_acc['model'])} has the highest original clean-test accuracy ({percent(highest_acc['clean_accuracy'])}%). "
                  + ("The two rankings agree here." if lowest_var["model"] == highest_acc["model"] else "The two rankings differ here."))
    if tree["depth"] is None:
        depth_protocol = ("The selected tree already has unlimited depth, so a deeper condition cannot be built. "
                          "The comparison therefore uses depths 2, 8, and None, with the selected minimum leaf size and 100 trees per bagging ensemble.")
    else:
        shallow_depth = 1 if tree["depth"] == 2 else 2
        depth_protocol = (f"The depth study uses {shallow_depth}, selected depth {tree['depth']}, and unlimited depth. "
                          "The minimum leaf size stays fixed, with 100 trees per bagging ensemble.")
    pages.append({"title": "Discussion and reproducibility", "full": [final_table], "left": [
        h("18. Overall comparison"),
        p(conclusion),
        p(f"{label(most_drop['model'])} is also most sensitive to the mask by accuracy drop. Stability under training resampling and resistance to occlusion are therefore different properties. The depth comparison on this page also shows why variance alone cannot rank accuracy."),
        p("Table 8 combines the original models' clean and masked accuracies with bias and variance across 30 refits. The refits' mean accuracy and SD appear separately in Table 6."),
        h("19. Reproduction and validation"),
        p("README commands reproduce the split, 30 resamples, model selection, tables, figures, and report. results/summary.json contains scores, timings, OOB coverage, and traces; results/predictions.npz contains class vectors, votes, sample weights, and all 30 refits."),
        p("Checks verify the toy update, vector normalization, OOB eligibility, mask coordinates, identity counts, deterministic outputs, and the bias-plus-variance identity. Ensemble metrics come from the manual implementations."),
        h("20. Scope limits"),
        p("The test faces belong to known training identities, so this study concerns new images of those people. It covers one split and one central zero mask. Each requested size has one fresh-fit timing, without repeated benchmark measurements.")], "right": [
        h("Choosing comparison depths"),
        p(depth_protocol),
        p(variance_counterexample),
        h("References"),
        p("[1] scikit-learn. fetch_olivetti_faces API and Olivetti faces dataset documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html"),
        p("[2] scikit-learn. DecisionTreeClassifier API. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"),
        p("[3] L. Breiman. Bagging Predictors. Machine Learning 24, 123-140, 1996. https://doi.org/10.1007/BF00058655"),
        p("[4] J. Zhu, H. Zou, S. Rosset, and T. Hastie. Multi-class AdaBoost. Statistics and Its Interface 2, 349-360, 2009. https://doi.org/10.4310/SII.2009.v2.n3.a8"),
        h("Environment"),
        p(f"Python {meta.get('python', 'not recorded')}; platform {meta.get('platform', 'not recorded')}; machine {meta.get('machine', 'not recorded')}. Package versions: " + "; ".join(f"{k} {v}" for k, v in meta.get('environment', {}).items()) + "."),
        p(f"The experiment run took {number(meta.get('total_experiment_wall_seconds'), 1)} seconds. Table 4 times exclude evaluation and report generation. Configurations and run fingerprints identify the inputs for cached refits.")]})
    return pages


def latex_escape(value):
    mappings = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mappings.get(c, c) for c in str(value))


def latex_paragraph(value):
    # hyperref's URL command supplies legal breakpoints in long API references.
    parts = re.split(r"(https?://\S+)", value)
    return "".join(r"\url{" + part + "}" if part.startswith(("https://", "http://")) else latex_escape(part)
                   for part in parts)


def tex_blocks(blocks, full=False):
    lines = []
    for b in blocks:
        if b["kind"] == "heading":
            lines += [r"\subsection*{" + latex_escape(b["text"]) + "}"]
        elif b["kind"] == "paragraph":
            lines += [latex_paragraph(b["text"]), ""]
        elif b["kind"] == "equation":
            lines += [b["latex"], ""]
        elif b["kind"] == "figure":
            maxh = b["height"] / 72
            lines += [r"{\centering", r"\IfFileExists{figures/" + b["name"] + r"_compact.pdf}{\def\reportfigure{figures/" + b["name"] + r"_compact.pdf}}{\def\reportfigure{../figures/" + b["name"] + r".pdf}}",
                      r"\includegraphics[width=0.98\textwidth,height=" + f"{maxh:.3f}" + r"in,keepaspectratio]{\reportfigure}",
                      r"\par\small " + latex_escape(b["caption"]), r"\par}", r"\vspace{4pt}"]
        else:
            width = r"0.99\textwidth" if full else r"\columnwidth"
            lines += [r"{\centering", r"\sbox{\reporttablebox}{%", r"\begin{tabular}{" + "l" * len(b["headers"]) + "}", r"\toprule",
                      " & ".join(latex_escape(v) for v in b["headers"]) + r" \\", r"\midrule"]
            lines += [" & ".join(latex_escape(v) for v in row) + r" \\" for row in b["rows"]]
            lines += [r"\bottomrule", r"\end{tabular}}",
                      r"\ifdim\wd\reporttablebox>" + width,
                      r"\resizebox{" + width + r"}{!}{\usebox{\reporttablebox}}",
                      r"\else\usebox{\reporttablebox}\fi",
                      r"\par\small " + latex_escape(b["caption"]), r"\par}", r"\vspace{4pt}"]
    return lines


def write_tex(path, pages, metadata):
    name = metadata.get("student_name") or "Student name not provided"
    roll = metadata.get("roll_number") or "not provided"
    repo = metadata.get("repository_url") or "not provided"
    commit = metadata.get("final_commit_hash") or "not provided"
    seed_type = "Provisional" if metadata.get("seed_is_provisional", True) else "Roll-derived"
    lines = [r"\documentclass[10pt,twocolumn,letterpaper]{article}",
             r"\usepackage[pagenumbers]{cvpr}", r"\usepackage[T1]{fontenc}", r"\usepackage{times}", r"\usepackage{graphicx}",
             r"\usepackage{booktabs}", r"\usepackage{amsmath,amssymb}", r"\usepackage{enumitem}",
             r"\usepackage[breaklinks=true,colorlinks=true,allcolors=blue]{hyperref}",
             r"\def\paperID{Task99}", r"\def\confName{CVPR}", r"\def\confYear{2026}",
             r"\setlength{\parskip}{2pt}", r"\setlength{\textfloatsep}{5pt}",
             r"\newsavebox{\reporttablebox}",
             r"\title{Bagging, Boosting, and Bias-Variance on Face Recognition}",
             r"\author{" + latex_escape(name) + r"; roll number: " + latex_escape(roll) + r"\\" +
             latex_escape(f"{seed_type} seed: {metadata.get('seed', 99)}; repository: {repo}; commit: {commit}") + "}",
             r"\begin{document}", r"\maketitle"]
    for i, page in enumerate(pages):
        if i:
            lines += [r"\clearpage", r"\twocolumn[{%", r"\section*{" + latex_escape(page["title"]) + "}"]
            lines += tex_blocks(page["full"], full=True)
            lines += [r"\vspace{3pt}}]"]
        else:
            lines += [r"\section*{" + latex_escape(page["title"]) + "}"]
        lines += tex_blocks(page["left"])
        # In two-column mode, newpage advances to the right-hand column. The
        # document model deliberately groups related sections by column so a
        # short page cannot leave a paragraph fragment above an empty column.
        lines += [r"\newpage"]
        lines += tex_blocks(page["right"])
    lines += [r"\end{document}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--source-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = args.summary or root / "results" / "summary.json"
    s = json.loads(summary.read_text(encoding="utf-8"))
    pages = build_pages(s)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    template_root = Path(__file__).resolve().parents[1] / "report"
    for template_name in ("cvpr.sty", "ieeenat_fullname.bst"):
        source = template_root / template_name
        target = out / template_name
        if source.is_file() and source.resolve() != target.resolve() and not target.exists():
            shutil.copyfile(source, target)
    write_tex(out / "report.tex", pages, s["metadata"])
    print(json.dumps({"report_pages": len(pages), "source": str(out / "report.tex")}, indent=2))


if __name__ == "__main__":
    main()
