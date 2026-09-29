"""
Train on 1.5M synthetic parts, test on a FRESH 1.5M synthetic parts.
Reuses the margin=0.75 already validated at this scale in earlier work
(5-fold CV on smaller data underestimated tail risk; 0.75 was the
margin that held) -- not re-derived here, ponytail: re-running CV would
cost ~10x compute for a number we already have.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

CHECKPOINTS = ["0h", "24h", "96h", "168h"]
HOURS = {"0h": 0, "24h": 24, "96h": 96, "168h": 168}
MAD_SCALE = 0.6745
MARGIN = 0.75  # established at 300k-1.5M scale previously; see docstring
PARAM_ANOMALY_CORR = 0.75

PARAMS = {
    "IDDQ_uA": dict(base=(5, 15), var=0.08, nrate=(0.004, 0.015), arate=(0.08, 0.20),
                    aacc=(0.0008, 0.0035), nnoise=0.25, anoise=0.8, inc=True, limit=80.0, floor=0.1),
    "leakage_uA": dict(base=(2, 8), var=0.08, nrate=(0.01, 0.05), arate=(0.15, 0.35),
                        aacc=(0.002, 0.006), nnoise=0.3, anoise=1.0, inc=True, limit=25.0, floor=0.1),
    "Vt_V": dict(base=(0.45, 0.65), var=0.03, nrate=(0.00003, 0.00015), arate=(0.0008, 0.0025),
                 aacc=(0.00001, 0.00004), nnoise=0.004, anoise=0.015, inc=True, limit=0.9, floor=0.1),
    "prop_delay_ns": dict(base=(1.5, 4), var=0.06, nrate=(0.001, 0.004), arate=(0.008, 0.025),
                           aacc=(0.0004, 0.0012), nnoise=0.04, anoise=0.12, inc=True, limit=16.0, floor=0.2),
    "supply_mA": dict(base=(20, 35), var=0.06, nrate=(0.008, 0.04), arate=(0.15, 0.4),
                       aacc=(0.0015, 0.005), nnoise=0.4, anoise=1.3, inc=True, limit=60.0, floor=1.0),
    "freq_MHz": dict(base=(80, 120), var=0.04, nrate=(-0.015, -0.004), arate=(-0.30, -0.10),
                      aacc=(-0.006, -0.002), nnoise=0.4, anoise=1.2, inc=False, limit=60.0, floor=10.0),
}


def gen_dataset(n_lots, parts_per_lot, anomaly_rate, seed):
    """Vectorized generator (needed at 1.5M scale on 1 CPU). Same model as
    the validated per-part version: guarantees every ground-truth anomaly
    manifests on >=1 parameter (the fix found necessary during earlier
    300k-scale testing -- without it, ~0.02% of anomalies are silent/
    undetectable by construction)."""
    rng = np.random.default_rng(seed)
    n = n_lots * parts_per_lot
    lot_id = np.repeat(np.arange(n_lots), parts_per_lot)
    is_anom = rng.random(n) < anomaly_rate

    # per-parameter anomaly flags, with the "force at least one" guarantee
    flags = rng.random((n, len(PARAMS))) < PARAM_ANOMALY_CORR
    flags &= is_anom[:, None]
    none_flagged = is_anom & ~flags.any(axis=1)
    if none_flagged.any():
        forced_col = rng.integers(0, len(PARAMS), size=none_flagged.sum())
        rows = np.where(none_flagged)[0]
        flags[rows, forced_col] = True

    cols = {"Lot_ID": lot_id, "Anomaly_GroundTruth": is_anom.astype(int)}
    for j, (pname, c) in enumerate(PARAMS.items()):
        lot_baseline = rng.uniform(*c["base"], size=n_lots)[lot_id]
        anom = flags[:, j]
        start = np.maximum(c["floor"], rng.normal(lot_baseline, lot_baseline * c["var"]))
        rate = np.where(anom, rng.uniform(*c["arate"], size=n), rng.uniform(*c["nrate"], size=n))
        acc = np.where(anom, rng.uniform(*c["aacc"], size=n), 0.0)
        noise_scale = np.where(anom, c["anoise"], c["nnoise"])
        for cp in CHECKPOINTS:
            t = HOURS[cp]
            val = start + rate * t + acc * (t ** 1.5) + rng.normal(0, noise_scale)
            val = np.clip(val, c["floor"], c["limit"] * 0.98) if c["inc"] else np.maximum(val, c["limit"] * 1.02)
            cols[f"{pname}_{cp}"] = val
    return pd.DataFrame(cols)


def modz(values):
    med = np.median(values)
    mad = np.median(np.abs(values - med)) or 1e-6
    return MAD_SCALE * (values - med) / mad


def module_a_signal(df):
    sig = np.zeros(len(df))
    lot_groups = df.groupby("Lot_ID").indices  # compute once, reused for every (param, checkpoint)
    for pname in PARAMS:
        for cp in CHECKPOINTS:
            col = df[f"{pname}_{cp}"].to_numpy()
            for idx in lot_groups.values():
                z = np.abs(modz(col[idx]))
                np.maximum.at(sig, idx, z)
    return sig


def train_module_b(df):
    models = {}
    for pname, c in PARAMS.items():
        sign = 1.0 if c["inc"] else -1.0
        v0, v24, v168 = df[f"{pname}_0h"].to_numpy(), df[f"{pname}_24h"].to_numpy(), df[f"{pname}_168h"].to_numpy()
        X = np.column_stack([v0, v24, (v24 - v0) / 24.0])
        model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10,
                                       max_samples=0.1, random_state=42, n_jobs=1)
        model.fit(X, v168)
        normal = df["Anomaly_GroundTruth"].to_numpy() == 0
        slopes = sign * (v168[normal] - v24[normal]) / (168 - 24)
        scale = max(np.percentile(np.abs(slopes), 95), 1e-9)
        models[pname] = (model, sign, scale)
    return models


def module_b_signal(df, models):
    sig = np.zeros(len(df))
    for pname, (model, sign, scale) in models.items():
        v0, v24 = df[f"{pname}_0h"].to_numpy(), df[f"{pname}_24h"].to_numpy()
        X = np.column_stack([v0, v24, (v24 - v0) / 24.0])
        pred = model.predict(X)
        severity = np.abs(sign * (pred - v24) / (168 - 24)) / scale
        sig = np.maximum(sig, severity)
    return sig


def evaluate(df, signal, review_th, reject_th, label):
    y = df["Anomaly_GroundTruth"].to_numpy()
    tier = np.where(signal >= reject_th, "REJECT", np.where(signal >= review_th, "REVIEW", "PASS"))
    missed = int(((tier == "PASS") & (y == 1)).sum())
    total_anom = int((y == 1).sum())
    flagged_normal = int(((tier != "PASS") & (y == 0)).sum())
    total_normal = int((y == 0).sum())
    print(f"[{label}] n={len(df)} anomalies={total_anom} | missed={missed}/{total_anom} | "
          f"normal-flagged={flagged_normal}/{total_normal} ({100*flagged_normal/total_normal:.2f}%)")
    return missed


if __name__ == "__main__":
    t0 = time.time()
    N_LOTS, PARTS_PER_LOT, RATE = 1500, 1000, 0.03  # 1.5M parts

    print("Generating + training on 1.5M parts...")
    df_train = gen_dataset(N_LOTS, PARTS_PER_LOT, RATE, seed=1)
    models = train_module_b(df_train)
    sig_a = module_a_signal(df_train)
    sig_b = module_b_signal(df_train, models)
    combined_train = np.maximum(sig_a, sig_b)

    anomaly_min = combined_train[df_train["Anomaly_GroundTruth"] == 1].min()
    reject_th = anomaly_min * 0.98
    review_th = anomaly_min * MARGIN
    print(f"review_th={review_th:.3f} reject_th={reject_th:.3f} (margin={MARGIN}) [{time.time()-t0:.0f}s elapsed]")
    evaluate(df_train, combined_train, review_th, reject_th, "TRAIN (sanity check)")

    print("\nGenerating FRESH 1.5M test set (new seed)...")
    df_test = gen_dataset(N_LOTS, PARTS_PER_LOT, RATE, seed=999)
    sig_a_t = module_a_signal(df_test)
    sig_b_t = module_b_signal(df_test, models)
    combined_test = np.maximum(sig_a_t, sig_b_t)
    evaluate(df_test, combined_test, review_th, reject_th, "FRESH TEST")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
