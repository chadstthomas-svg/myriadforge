"""Learn auditable Boolean strings from the SLURP text labels, then compile them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

from build import compile_policy
from policy import LEXEME, Policy


def phrases(s):
    words = LEXEME.findall(s.casefold())
    return words + [" ".join(pair) for pair in zip(words,words[1:])]


def splits(cache):
    seen = set()
    data = {key: [] for key in ("train","calibrate","select","test")}
    for filename, name in (("train","train"),("devel","dev"),("test","test")):
        with (cache/(filename+".jsonl")).open(encoding="utf8") as src:
            for line in src:
                r = json.loads(line)
                key = " ".join(r["sentence"].lower().split())
                if key in seen:
                    continue
                seen.add(key)
                split = name
                if name == "dev":
                    split = "calibrate" if hashlib.sha256(key.encode()).digest()[0]%2 else "select"
                data[split].append((r["sentence"],r["scenario"],r["intent"]))
    return data


def boolean_tree(tree, feature_names, leaf_predictions, accepted):
    rules, cat, act = {}, {}, {}
    def visit(i, path):
        left, right = tree.children_left[i], tree.children_right[i]
        if left == right:
            name = "leaf_%d" % i
            rules[name] = " AND ".join(path)
            category, action = leaf_predictions[i]
            cat.setdefault(category, []).append("@"+name)
            if i in accepted:
                act.setdefault(action, []).append("@"+name)
            return
        term = json.dumps(feature_names[tree.feature[i]],ensure_ascii=False)
        visit(left, path + ["NOT "+term])
        visit(right, path + [term])
    visit(0,[])
    if not act:
        raise ValueError("Calibration produced no qualifying leaves")
    rules["may_propose"] = " OR ".join(name for group in act.values() for name in group)
    return dict(rules=rules, answers={"may_propose_action":"@may_propose"},
                categories={k:" OR ".join(v) for k,v in cat.items()},
                actions={k:" OR ".join(v) for k,v in act.items()})


def score(leaves, rows, predictions, accepted):
    categories = actions = correct_actions = 0
    for leaf, (_,category,action) in zip(leaves,rows):
        cat, tool = predictions[leaf]
        categories += int(cat == category)
        if leaf in accepted:
            actions += 1
            correct_actions += int(tool == action)
    return dict(examples=len(rows),category_accuracy=categories/len(rows),
                proposals=actions,correct_proposals=correct_actions,
                coverage=actions/len(rows),precision=correct_actions/max(actions,1))


def run(cache, directory):
    data = splits(cache)
    v = CountVectorizer(analyzer=phrases, binary=True, min_df=5, dtype=np.float32)
    train = data["train"]
    matrices = {"train":v.fit_transform([r[0] for r in train])}
    for name in ("calibrate","select","test"):
        matrices[name] = v.transform(r[0] for r in data[name])
    names = v.get_feature_names_out()
    trials = []
    for capacity in [8,16,32,64]:
        tree = DecisionTreeClassifier(max_leaf_nodes=capacity, max_depth=12, min_samples_leaf=18,
                                      random_state=131)
        tree.fit(matrices["train"], [r[1:] for r in train])
        memberships = tree.apply(matrices["train"])
        predictions = {}
        for leaf in set(memberships):
            subset = [train[j] for j in np.flatnonzero(memberships==leaf)]
            predictions[leaf] = tuple(Counter(r[c] for r in subset).most_common(1)[0][0] for c in (1,2))
        tally = {}
        for leaf,row in zip(tree.apply(matrices["calibrate"]),data["calibrate"]):
            counts = tally.setdefault(leaf,[0,0])
            counts[0] += int(predictions[leaf][1] == row[2])
            counts[1] += 1
        accepted = {i for i,(right,n) in tally.items() if n>=10 and right/n>=0.90}
        sel = score(tree.apply(matrices["select"]),data["select"],predictions,accepted)
        if accepted:
            spec = boolean_tree(tree.tree_,names,predictions,accepted)
            trials.append((dict(max_leaves=capacity,evidence_literals=len(Policy(spec).literals),
                                selection=sel),spec,tree,predictions,accepted))
    if not trials:
        raise RuntimeError("No candidate met calibration criteria")
    # Lock selection before reading test labels. This is a developmental size/quality tradeoff.
    pool = [t for t in trials if t[0]["selection"]["precision"]>=0.9] or trials
    best_cat = max(t[0]["selection"]["category_accuracy"] for t in pool)
    near = [t for t in pool if t[0]["selection"]["category_accuracy"] >= best_cat-0.03]
    best_coverage = max(t[0]["selection"]["coverage"] for t in near)
    chosen = min((t for t in near if t[0]["selection"]["coverage"] >= 0.9*best_coverage),
                 key=lambda t:t[0]["max_leaves"])
    info,spec,tree,predictions,accepted = chosen
    program = Policy(spec)
    test = data["test"]
    leaves = tree.apply(matrices["test"])
    expected_rows = []
    for leaf,(sentence,category,action) in zip(leaves,test):
        decision = program.result(program.reference(program.text_atoms(sentence)))
        cat, tool = predictions[leaf]
        truth = dict(answers={"may_propose_action":leaf in accepted},category=cat,
                     action=tool if leaf in accepted else None)
        if decision != truth:
            raise AssertionError("Exported Boolean strings disagree with training tree")
        expected_rows.append(dict(text=sentence,expected=truth,category_label=category,action_label=action))
    directory.mkdir(exist_ok=True,parents=True)
    source = directory/"learned-support.json"
    source.write_text(json.dumps(spec,indent=2)+"\n",encoding="utf8")
    compile_policy(source,directory/"bundles"/"learned","gates")
    (directory/"selection.json").write_text(json.dumps(dict(
        trials=[t[0] for t in trials],selected=info["max_leaves"],
        split_counts={k:len(v) for k,v in data.items()},
        test=score(leaves,test,predictions,accepted)),indent=2)+"\n")
    (cache/"boolean-learned-test.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in expected_rows),encoding="utf8")
    print(info,score(leaves,test,predictions,accepted))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--directory",type=Path,default=Path(__file__).parent)
    args=p.parse_args()
    run(args.cache,args.directory)
