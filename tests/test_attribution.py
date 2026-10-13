import numpy as np

from brain_lora.analysis.attribution import BIT, FULL, MODS, derive, parcel_groups, shapley, subsets


def test_shapley_is_additive_and_symmetric():
    rng = np.random.default_rng(0)
    subs = subsets()
    masks = np.array([sum(BIT[m] for m in s) for s in subs])
    pred = rng.normal(size=(8, 3, 4))
    phi, f = shapley(pred, masks)
    assert np.allclose(f(0) + sum(phi.values()), f(FULL))
    # an additive game gives each player its own marginal
    gain = {m: rng.normal(size=(3, 4)) for m in MODS}
    additive = np.stack([sum(gain[m] for m in s) if s else np.zeros((3, 4)) for s in subs])
    phi, _ = shapley(additive, masks)
    assert all(np.allclose(phi[m], gain[m]) for m in MODS)


def test_accuracy_contributions_sum_to_r():
    rng = np.random.default_rng(1)
    S, T, P, K = 2, 400, 3, 4
    parts = rng.normal(size=(K, S, T, P))
    f = parts.sum(0)
    y = f + rng.normal(size=(S, T, P))
    tot = {"n": np.full(S, T, float), "sy": y.sum(1), "syy": (y * y).sum(1), "sf": f.sum(1),
           "sff": (f * f).sum(1), "sfy": (f * y).sum(1), "sx": parts.sum(2),
           "sxx": (parts * parts).sum(2), "sxf": (parts * f).sum(2), "sxy": (parts * y).sum(2)}
    out = derive(tot, series="abcd")
    assert np.allclose(sum(out["A_" + k] for k in "abcd"), out["r"])
    assert np.allclose(sum(out["S_" + k] for k in "abcd"), 1.0)


def test_region_labels_cover_the_parcellation():
    groups, labels = parcel_groups()
    assert len(labels) == 1000
    assert groups["Whole cortex"].all()
    assert groups["Auditory cortex"].sum() > 0 and groups["Early visual (V1/V2)"].sum() > 0
