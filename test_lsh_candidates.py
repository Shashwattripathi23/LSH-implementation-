import itertools

import numpy as np

import lsh_candidates as lc
from tune_lsh import exact_jaccard


def synthetic_ratings(n_users, n_movies, n_planted, seed, k_range=(50, 500)):
    """Popularity-skewed random users plus planted pairs with Jaccard in (0.5, 0.9)."""
    rng = np.random.default_rng(seed)
    pop = 1.0 / (np.arange(n_movies) + 10.0) ** 0.8
    pop /= pop.sum()
    sizes = np.exp(rng.uniform(*np.log(k_range), size=n_users)).astype(int)
    sets = [rng.choice(n_movies, size=k, replace=False, p=pop) + 1 for k in sizes]
    users = rng.permutation(n_users)[: 2 * n_planted].reshape(-1, 2)
    for a, c in users:
        A = sets[a]
        n = len(A)
        J = rng.uniform(0.5, 0.9)
        m = int(round(2 * n * J / (1 + J)))
        outside = np.setdiff1d(np.arange(1, n_movies + 1), A)
        sets[c] = np.concatenate([rng.choice(A, m, replace=False), rng.choice(outside, n - m, replace=False)])
    uid = np.repeat(np.arange(1, n_users + 1), [len(s) for s in sets])
    mid = np.concatenate(sets)
    return np.column_stack([uid, mid, rng.integers(1, 6, size=len(mid))]).astype(np.int32)


def brute_force_jaccard(M):
    X = M.astype(np.int32)
    inter = (X @ X.T).toarray()
    sizes = np.diff(M.indptr)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return np.triu(J, 1)


def test_assignment_example():
    ratings = np.array([[1, 2, 5], [1, 3, 4], [1, 5, 3], [2, 1, 5], [2, 2, 4], [2, 3, 3]])
    M = lc.build_user_movie_matrix(ratings)
    assert M.shape == (3, 6)
    assert list(M[1].indices) == [2, 3, 5] and list(M[2].indices) == [1, 2, 3]
    assert exact_jaccard(M, np.array([1]), np.array([2]))[0] == 0.5


def test_duplicate_ratings_collapse():
    M = lc.build_user_movie_matrix(np.array([[1, 4, 5], [1, 4, 3], [1, 2, 1]]))
    assert M.nnz == 2 and set(M.data) == {1}


def test_minhash_matches_definition():
    rng = np.random.default_rng(0)
    dense = rng.random((30, 25)) < 0.2
    dense[[0, 7, 13]] = False
    M = lc.sparse.csr_matrix(dense.astype(np.int8))
    h = 40
    sig = lc.minhash_signatures(M, h, np.random.default_rng(5))
    rng2 = np.random.default_rng(5)
    perms = [rng2.permutation(25) for _ in range(h)]
    for u in range(30):
        movies = np.flatnonzero(dense[u])
        for i in range(h):
            expected = perms[i][movies].min() if len(movies) else 25
            assert sig[i, u] == expected


def test_signature_agreement_estimates_jaccard():
    ratings = synthetic_ratings(400, 1500, 40, seed=1)
    M = lc.build_user_movie_matrix(ratings)
    sig = lc.minhash_signatures(M, 1000, np.random.default_rng(2))
    sigT = np.ascontiguousarray(sig.T)
    rng = np.random.default_rng(3)
    lo = rng.integers(1, 401, 300)
    hi = rng.integers(1, 401, 300)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    est = lc.agreeing_rows(sigT, lo, hi) / 1000
    assert np.abs(est - exact_jaccard(M, lo, hi)).max() < 0.06


def test_band_buckets_match_dict_grouping():
    rng = np.random.default_rng(4)
    band = rng.integers(0, 3, size=(3, 500)).astype(np.int16)
    order, starts, sizes = lc.band_buckets(band)
    got = {frozenset(order[s:s + k]) for s, k in zip(starts, sizes)}
    groups = {}
    for col in range(500):
        groups.setdefault(tuple(band[:, col]), set()).add(col)
    expected = {frozenset(g) for g in groups.values() if len(g) > 1}
    assert got == expected


def test_pair_generators_match_combinations():
    old = lc.PAIR_CHUNK
    lc.PAIR_CHUNK = 7
    try:
        members = np.array([9, 3, 5, 1, 8, 2, 7, 4, 6, 0], dtype=np.int32)
        starts, sizes = np.array([0, 2, 6]), np.array([2, 4, 4])
        got = set()
        for lo, hi in lc._small_bucket_pairs(members, starts, sizes):
            assert np.all(lo < hi)
            got |= set(zip(lo.tolist(), hi.tolist()))
        expected = set()
        for s, k in zip(starts, sizes):
            expected |= {tuple(sorted(p)) for p in itertools.combinations(members[s:s + k].tolist(), 2)}
        assert got == expected

        bucket = np.random.default_rng(0).permutation(23).astype(np.int32)
        got = []
        for lo, hi in lc._big_bucket_pairs(bucket):
            got += list(zip(lo.tolist(), hi.tolist()))
        assert sorted(got) == list(itertools.combinations(range(23), 2))
    finally:
        lc.PAIR_CHUNK = old


def _small_dataset():
    ratings = synthetic_ratings(3000, 3000, 150, seed=7)
    keep = ~np.isin(ratings[:, 0], [17, 123])
    return lc.build_user_movie_matrix(ratings[keep])


def test_output_format_and_empty_users():
    M = _small_dataset()
    pairs, sim, st = lc.find_candidate_pairs(M, seed=1, h=100, r=4, verbose=False)
    assert pairs.dtype == np.int32 and pairs.shape[1] == 2
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    assert np.all(np.diff(sim) <= 0)
    assert np.all(sim >= lc.default_min_sig_sim(100) - 1e-6)
    assert not np.isin(pairs, [0, 17, 123]).any()
    assert st["candidates"] == len(pairs)


def test_seed_determinism():
    M = _small_dataset()
    p1, _, _ = lc.find_candidate_pairs(M, seed=3, h=60, r=4, verbose=False)
    p2, _, _ = lc.find_candidate_pairs(M, seed=3, h=60, r=4, verbose=False)
    p3, _, _ = lc.find_candidate_pairs(M, seed=4, h=60, r=4, verbose=False)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


def test_postponed_buckets_give_same_result():
    M = _small_dataset()
    normal, _, _ = lc.find_candidate_pairs(M, seed=2, h=60, r=3, max_bucket=10**9, verbose=False)
    postponed, _, st = lc.find_candidate_pairs(M, seed=2, h=60, r=3, max_bucket=2, verbose=False)
    assert st["big_buckets"] > 0 and st["big_buckets_done"] == st["big_buckets"]
    assert set(map(tuple, normal)) == set(map(tuple, postponed))


def test_time_limit_stops_early_with_valid_output():
    M = _small_dataset()
    pairs, sim, st = lc.find_candidate_pairs(M, seed=1, h=60, r=3, time_limit=0, verbose=False)
    assert st["bands_done"] < st["b"]
    assert np.all(pairs[:, 0] < pairs[:, 1]) and len(pairs) == len(sim)


def test_recall_follows_s_curve():
    M = _small_dataset()
    J = brute_force_jaccard(M)
    true_lo, true_hi = np.nonzero(J > lc.THRESHOLD)
    truth = set(zip(true_lo.tolist(), true_hi.tolist()))
    h, r = 100, 5
    b = h // r
    expected = lc.s_curve(J[true_lo, true_hi], r, b).sum()
    found = []
    for seed in range(6):
        pairs, _, _ = lc.find_candidate_pairs(M, seed=seed, h=h, r=r, verbose=False)
        found.append(len(truth & set(map(tuple, pairs.tolist()))))
    sd = np.sqrt((lc.s_curve(J[true_lo, true_hi], r, b) * (1 - lc.s_curve(J[true_lo, true_hi], r, b))).sum())
    print(f"true pairs {len(truth)}, expected found {expected:.1f} (sd {sd:.1f}), found per seed {found}")
    assert abs(np.mean(found) - expected) < 3 * sd / np.sqrt(len(found)) + 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
