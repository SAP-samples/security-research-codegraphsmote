import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import re
import csv
import json
import math

from collections import defaultdict


def aggregate(metrics):
    ret = {}
    for full_key, metrics in metrics.items():
        if len(metrics["MCC"]) < 2:
            continue # filter single runs
        if "EDGEDROP" in full_key:
            continue
        ret[full_key] = {}
        if "MCC" not in metrics:
            metrics["MCC"] = [(tp*tn - fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-12) \
                for (tp, tn, fp, fn) in zip(metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"])]
        for key in metrics.keys():
            mu = sum(metrics[key]) / len(metrics[key])
            ret[full_key][key + "_mu"] = mu
            ret[full_key][key + "_std"] = math.sqrt(sum((x - mu)**2 for x in metrics[key]) / len(metrics[key]))
            ret[full_key][key + "_min"] = min(metrics[key])
            ret[full_key][key + "_max"] = max(metrics[key])
            ret[full_key][key + "_full"] = metrics[key]
        ret[full_key]["count"] = len(metrics["MCC"])
    return ret


def get_metrics(metrics):
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for key, vals in metrics.items():
        key = key.split('__')[0]
        dataset = "_".join(key.split("_")[1:-2])
        percentage = re.findall(r"\d+", dataset)[-1]
        dataset = dataset.replace(percentage, "")
        # dataset, percentage = dataset[:-2], dataset[-2:]
        dataset += "_" + key.split("_")[-1]
        for key, val in vals.items():
            res[dataset][int(percentage)][key] = val
    return res


def make_key_flatten(l):
    ret = []
    if type(l) is list:
        for d in l:
            ret.extend(make_key_flatten(d))
    elif type(l) is dict:
        for k, v in l.items():
            if type(v) is dict:
                ret.extend([str(k) + "_" + str(x) for x in make_key_flatten(v)])
            if type(v) is list:
                # assume contains serializable
                ret.extend([str(k) + ":" + str(x) for x in v])
            else:
                # assume serializable
                ret.append(str(k) + ":" + str(v))
    else:
        raise ValueError(f"Type {type(l)} of {l} unknown")
    return ret


def make_key(l):
    strs = []
    dicts = []
    for val in l:
        if isinstance(val, dict):
            dicts.append(val)
        elif isinstance(val, str):
            strs.append(val)
        else:
            raise ValueError(f"Type {type(val)} of {val} unknown")
    strs_s = "_".join(sorted(strs))
    dicts_s = "__".join(sorted(make_key_flatten(dicts)))

    return strs_s + "__" + dicts_s


def plot(metrics, key):
    Xs = {}
    Ys = {}
    Y1s = {}
    Y2s = {}
    for dataset, runs in metrics.items():
        X = list(sorted(runs.keys()))
        if runs[X[0]].get(key+"_mu") is None:
            print(f"Skipping {dataset} for key {key}")
            continue
        Y = np.asarray([runs[x][key+"_mu"] for x in X])
        Ystd = np.asarray([runs[x][key+"_std"] for x in X])
        Y1 = Y - Ystd
        Y2 = Y + Ystd

        Xs[dataset] = X
        Ys[dataset] = Y
        Y1s[dataset] = Y1
        Y2s[dataset] = Y2

        matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams.update({'font.size': 20})
    for runset in ["PATCHDB", "REVEAL3", "QEMU_FFMPEG"]:
        if not any(runset in dataset for dataset in Xs.keys()):
            continue
        plt.clf()
        plt.rc("xtick", labelsize=36)
        plt.rc("ytick", labelsize=36)
        plt.rc("axes", labelsize=38)
        fig, ax = plt.subplots(figsize =(12, 8))
        ylim_min = 1
        ylim_max = 0
        xlim_min = 100
        xlim_max = 0

        def labelize(s):
            mapping = {
                "DOWNSAMPLED": "Downsampled",
                "SARD": "SARD",
                "SMOTE": "CodeGraphSMOTE",
                "NODEDROP": "Node-Dropping",
                "EDGEDROP": "Edge-Dropping"
            }
            for k, v in mapping.items():
                if k in s:
                    return v
            return s
        
        def labelize_key(s):
            mapping = {
                "BAcc": "Balanced Accuracy",
                "F1": "F1-Score",
                "MCC": "Matthews Correlation Coefficient ",
            }
            for k, v in mapping.items():
                if k in s:
                    return v
            return s

        for dataset in sorted(Xs.keys()):
            if runset not in dataset:
                continue
            ylim_min = min(ylim_min, min(Y1s[dataset]))
            ylim_max = max(ylim_max, max(Y2s[dataset]))
            xlim_min = min(xlim_min, min(Xs[dataset]))
            xlim_max = max(xlim_max, max(Xs[dataset]))
            def scale_auc(auc):
                return auc / (max(Xs[dataset])/100 - min(Xs[dataset])/100)
            auc = scale_auc(np.trapz(y = Ys[dataset], x=np.asarray(Xs[dataset])/100))
            auc_min = scale_auc(np.trapz(y = Y1s[dataset], x=np.asarray(Xs[dataset])/100))
            auc_max = scale_auc(np.trapz(y = Y2s[dataset], x=np.asarray(Xs[dataset])/100))
            auc_std = (auc_max - auc_min) / 2
            ax.plot(Xs[dataset], Ys[dataset], label=f"{labelize(dataset)} {auc:.3f} ± {auc_std:.3f}")
            # ax.plot(Xs[dataset], Ys[dataset], label=f"{labelize(dataset)}")
            ax.fill_between(Xs[dataset], Y1s[dataset], Y2s[dataset], alpha=.1)

        ax.set_ylim(ylim_min, ylim_max)
        # ax.set_ylim(0.49, 0.76)
        ax.set_xlim(xlim_min, xlim_max+2)
        ax.set_xlim(20, 100)
        ax.set_ylabel(labelize_key(key))
        ax.set_xlabel("Percentage of dataset")
        fig.subplots_adjust(bottom=0.15)
        # fig.legend(fontsize=20, loc="best")
        fig.savefig(f"results/plots/{runset}_{key}.pdf", bbox_inches="tight")

        figlegend = plt.figure()
        figlegend.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
        figlegend.savefig(f"results/plots/{runset}_legend.pdf", bbox_inches="tight")
        
        plt.close()


def remove_seed(d):
    if d.get("seed") is not None:
        del d["seed"]
    for k, v in d.items():
        if type(v) is dict:
            remove_seed(v)
def main():
    metrics = defaultdict(lambda : defaultdict(list))
    with open("results/results.csv", "r") as f:
        for row in csv.reader(f, delimiter=";"):
            for i in range(len(row)):
                if row[i][0] == "{":
                    # assume json
                    row[i] = json.loads(row[i])
                    remove_seed(row[i])
            if row[0] != "Test" or row[1] != "Checkpoint" or "epoch" in row[3] or row[4] != "baseline":
                continue
            full_key = make_key(row[:-1])
            for key, val in row[-1].items():
                metrics[full_key][key].append(val)
    
    aggregated = aggregate(metrics)
    metrics = get_metrics(aggregated)

    plot(metrics, "BBAcc")
    plot(metrics, "AP")
    plot(metrics, "BF1")


if __name__ == "__main__":
    if False:
        with plt.xkcd():
            main()
    else:
        main()