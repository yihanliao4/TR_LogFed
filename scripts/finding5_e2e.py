"""CPU-only post-processor for Finding 5 average end-to-end run time.

Reads the two profiler CSVs produced by efficiency_profile.py:
  revision_id/unified/efficiency/eff_timing.csv
  revision_id/unified/efficiency/eff_params.csv
and computes the average full federated run time for R in {30,50,100}
at 1/5/20 Mbps:

    avg_e2e(R, bw) = R * (avg_compute_per_round + analytic_round_trip(bw))

The compute term is the profiler's already-averaged per-round wall-clock
(fed_cross_round for non-IID, fed_iid_round for each per-dataset IID);
the RTT is the analytic upload+download of the FP32 model.

Writes finding5_e2e.csv in the same directory and prints a compact table.
No GPU / torch required.
"""
import os
import pandas as pd

EFF_DIR = "revision_id/unified/efficiency"
ROUNDS = [30, 50, 100]
BWS = [1, 5, 20]


def main():
    t = pd.read_csv(f"{EFF_DIR}/eff_timing.csv")
    p = pd.read_csv(f"{EFF_DIR}/eff_params.csv")

    rows = []
    # ---- cross-dataset non-IID (the deployment scenario in Finding 5) ----
    cross_t = t[(t["dataset"] == "non-IID") &
                (t["unit"] == "fed_cross_round")].set_index("model")
    cross_p = p[p["dataset"] == "non-IID"].set_index("model")
    for model in cross_t.index:
        comp = float(cross_t.loc[model, "seconds_per_unit"])
        for bw in BWS:
            rtt = float(cross_p.loc[model, f"rtt_{bw}Mbps_s"])
            for R in ROUNDS:
                rows.append((model, "non-IID", R, bw, round(comp, 3),
                             round(rtt, 3), round(R * (comp + rtt), 1)))

    # ---- per-dataset IID for completeness (Table VII) ----
    iid_t = t[t["unit"] == "fed_iid_round"].set_index(["dataset", "model"])
    for (ds, model), rr in iid_t.iterrows():
        comp = float(rr["seconds_per_unit"])
        pm = p[(p["model"] == model) & (p["dataset"] == ds)]
        if pm.empty:
            continue
        for bw in BWS:
            rtt = float(pm.iloc[0][f"rtt_{bw}Mbps_s"])
            for R in ROUNDS:
                rows.append((model, f"IID-{ds}", R, bw, round(comp, 3),
                             round(rtt, 3), round(R * (comp + rtt), 1)))

    out = pd.DataFrame(rows, columns=["model", "scenario", "rounds",
                                      "bandwidth_mbps", "avg_compute_per_round_s",
                                      "rtt_s", "avg_end_to_end_s"])
    out.to_csv(f"{EFF_DIR}/finding5_e2e.csv", index=False)

    # compact summary: non-IID at 20 Mbps (the headline Finding 5 numbers)
    print("\n=== Finding 5: average non-IID end-to-end @20 Mbps ===")
    sub = out[(out["scenario"] == "non-IID") & (out["bandwidth_mbps"] == 20)]
    piv = sub.pivot_table(index="model", columns="rounds",
                          values="avg_end_to_end_s")
    print(piv.round(1).to_string())
    print("\n(per-model per-round compute + RTT:")
    print(cross_t[["seconds_per_unit", "samples_per_sec"]].to_string())
    print("\nWrote:", f"{EFF_DIR}/finding5_e2e.csv")


if __name__ == "__main__":
    main()
