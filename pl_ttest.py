"""R5 statistics: aggregate pl_ablation.csv and run paired t-tests of the
soft-prompt variants (P=5, P=10) against the no-prompt baseline (P=0) on the
same (dataset, seed, setting) split. Also reports mean +/- std and the delta.
"""
import csv, argparse
from collections import defaultdict
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="revision_id/unified/pl_ablation.csv")
    a = ap.parse_args()

    # key: (dataset, seed, setting) -> {prompt: F1}
    d = defaultdict(dict)
    for r in csv.DictReader(open(a.csv)):
        d[(r["dataset"], r["seed"], r["setting"])][int(r["prompt_len"])] = float(r["F1"])

    datasets = sorted({k[0] for k in d})
    settings = ["central", "fed_iid", "fed_noniid"]
    print(f"{'dataset':11s} {'setting':11s} {'P=0':>9s} {'P=5':>9s} {'P=10':>9s}  "
          f"{'5vs0 t':>8s} {'p':>8s}  {'10vs0 t':>8s} {'p':>8s}")
    for ds in datasets:
        for s in settings:
            cells = {P: [] for P in (0, 5, 10)}
            for (dd, seed, st), mp in d.items():
                if dd == ds and st == s and all(P in mp for P in (0, 5, 10)):
                    for P in (0, 5, 10):
                        cells[P].append(mp[P])
            if not cells[0]:
                continue
            base = cells[0]
            def show(prompt):
                v = cells[prompt]
                delta = [v[i] - base[i] for i in range(len(base))]
                t, p = stats.ttest_rel(v, base)
                return (sum(v)/len(v), delta, t, p)
            m5 = show(5); m10 = show(10)
            mb = sum(base)/len(base)
            print(f"{ds:11s} {s:11s} {mb:9.4f} {m5[0]:9.4f} {m10[0]:9.4f}  "
                  f"{m5[2]:8.2f} {m5[3]:8.4f}  {m10[2]:8.2f} {m10[3]:8.4f}  "
                  f"meanΔ5={sum(m5[1])/len(m5[1]):+.4f} meanΔ10={sum(m10[1])/len(m10[1]):+.4f}")


if __name__ == "__main__":
    main()
