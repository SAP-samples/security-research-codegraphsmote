import csv
import json
import math
import statistics

from collections import defaultdict


def aggregate(metrics):
    ret = {}
    for full_key, metrics in metrics.items():
        ret[full_key] = {}
        if "MCC" not in metrics:
            metrics["MCC"] = [(tp*tn - fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-12) \
                for (tp, tn, fp, fn) in zip(metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"])]
        for key in metrics.keys():
            ret[full_key][key + "_mu"] = statistics.mean(metrics[key])
            ret[full_key][key + "_std"] = statistics.stdev(metrics[key]) if len(metrics[key]) > 1 else 0
            ret[full_key][key + "_max"] = max(metrics[key])
            ret[full_key][key + "_min"] = min(metrics[key])
        ret[full_key]["count"] = len(metrics["MCC"])
    return ret


def print_metrics(metrics):
    for key, vals in metrics.items():
        s = f"{key.split('__')[0]}:\n"
        if any(p in s for p in ["Train", "End"]):
            continue
        for key, val in vals.items():
            parts = key.split("_")
            if parts[-1] == "mu":
                key = "_".join(parts[:-1])
                if val <= 10:
                    s += f" {key}:\t{vals[key+'_mu']:.4f}\t± {vals[key+'_std']:.4f}\t[{vals[key+'_min']:.4f},\t{vals[key+'_max']:.4f}] \n"
                else:
                    s += f" {key}:\t{vals[key+'_mu']:.0f}\t± {vals[key+'_std']:.0f}\t[{vals[key+'_min']:.0f},\t{vals[key+'_max']:.0f}] \n"
            elif parts[-1] not in ["std", "max", "min"]:
                s += f" {key}:\t{vals[key]} \n"
        print(s)


def write_metrics(metrics):
    with open("results/results.txt", "w", encoding="utf-8") as f:
        for key, vals in metrics.items():
            full_key = key
            s = f"{key.split('__')[0]}:\n"
            for key, val in vals.items():
                parts = key.split("_")
                if parts[-1] == "mu":
                    key = "_".join(parts[:-1])
                    if val <= 10:
                        s += f" {key}:\t{vals[key+'_mu']:.4f}\t± {vals[key+'_std']:.4f}\t[{vals[key+'_min']:.4f},\t{vals[key+'_max']:.4f}] \n"
                    else:
                        s += f" {key}:\t{vals[key+'_mu']:.0f}\t± {vals[key+'_std']:.0f}\t[{vals[key+'_min']:.0f},\t{vals[key+'_max']:.0f}] \n"
                elif parts[-1] not in ["std", "max", "min"]:
                    s += f" {key}: {vals[key]} \n"
            s += f"full: {full_key} \n"
            f.write(s)
            f.write("\n")


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
            full_key = make_key(row[:-1])
            for key, val in row[-1].items():
                metrics[full_key][key].append(val)
    
    aggregated = aggregate(metrics)
    print_metrics(aggregated)
    write_metrics(aggregated)


if __name__ == "__main__":
    main()