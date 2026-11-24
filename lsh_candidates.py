import argparse
import math
import sys
import time

import numpy as np
from scipy import sparse

THRESHOLD = 0.5
DEFAULT_H = 600
DEFAULT_R = 5
DEFAULT_MAX_BUCKET = 1000
DEFAULT_MIN_SIG_SIM = None
SIG_SIM_Z = 3.5
DEFAULT_TIME_LIMIT = 900.0
PAIR_CHUNK = 2_000_000
SIM_CHUNK = 250_000


# Data Load
def build_user_movie_matrix(ratings):
    users = np.asarray(ratings[:, 0], dtype=np.int32)
    movies = np.asarray(ratings[:, 1], dtype=np.int32)
    shape = (int(users.max()) + 1, int(movies.max()) + 1)
    ones = np.ones(len(users), dtype=np.int8)
    M = sparse.csr_matrix((ones, (users, movies)), shape=shape)
    M.sum_duplicates()
    M.data[:] = 1
    return M


def load_user_movie_matrix(path):
    return build_user_movie_matrix(np.load(path, mmap_mode="r"))


def rated_users(M):
    return np.flatnonzero(np.diff(M.tocsr().indptr) > 0).astype(np.int32)


# Minhashing
def minhash_signatures(M, h, rng):
    C = M.tocsc()
    n_users, n_movies = C.shape
    dtype = np.int16 if n_movies < np.iinfo(np.int16).max else np.int32
    n_rated = len(rated_users(M))
    indptr, indices = C.indptr, C.indices
    sig = np.full((h, n_users), n_movies, dtype=dtype)
    for i in range(h):
        row = sig[i]
        left = n_rated
        for rank, movie in enumerate(np.argsort(rng.permutation(n_movies))):
            raters = indices[indptr[movie]:indptr[movie + 1]]
            new = raters[row[raters] == n_movies]
            if len(new):
                row[new] = rank
                left -= len(new)
                if left == 0:
                    break
    return sig


# Banding
def band_buckets(band):
    n = band.shape[1]
    order = np.lexsort(band)
    sorted_band = band[:, order]
    new = np.ones(n, dtype=bool)
    if n > 1:
        np.any(sorted_band[:, 1:] != sorted_band[:, :-1], axis=0, out=new[1:])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, n))
    keep = sizes > 1
    return order, starts[keep], sizes[keep]


def _small_bucket_pairs(members, starts, sizes):
    for k in np.unique(sizes):
        k = int(k)
        bucket_starts = starts[sizes == k]
        i, j = np.triu_indices(k, 1)
        per_chunk = max(1, PAIR_CHUNK // len(i))
        for c in range(0, len(bucket_starts), per_chunk):
            block = members[bucket_starts[c:c + per_chunk, None] + np.arange(k)]
            block.sort(axis=1)
            yield block[:, i].ravel(), block[:, j].ravel()


def _big_bucket_pairs(bucket):
    m = np.sort(bucket)
    k = len(m)
    t = 0
    while t < k - 1:
        rows = np.arange(t, min(k - 1, t + max(1, PAIR_CHUNK // (k - 1 - t))))
        counts = k - 1 - rows
        lo_idx = np.repeat(rows, counts)
        first = np.cumsum(counts) - counts
        hi_idx = np.arange(counts.sum()) - np.repeat(first, counts) + lo_idx + 1
        yield m[lo_idx], m[hi_idx]
        t = int(rows[-1]) + 1


def agreeing_rows(sigT, lo, hi):
    out = np.empty(len(lo), dtype=np.int32)
    for s in range(0, len(lo), SIM_CHUNK):
        e = s + SIM_CHUNK
        out[s:e] = np.count_nonzero(sigT[lo[s:e]] == sigT[hi[s:e]], axis=1)
    return out


# Candidate Pairs
def find_candidate_pairs(
    M,
    seed,
    h=DEFAULT_H,
    r=DEFAULT_R,
    b=None,
    max_bucket=DEFAULT_MAX_BUCKET,
    min_sig_sim=DEFAULT_MIN_SIG_SIM,
    time_limit=DEFAULT_TIME_LIMIT,
    verbose=True,
):
    t0 = time.perf_counter()
    deadline = None if time_limit is None else t0 + time_limit
    M = M.tocsr()
    sig = minhash_signatures(M, h, np.random.default_rng(seed))
    t_sig = time.perf_counter() - t0
    if verbose:
        print(f"[{t_sig:7.1f}s] signatures {sig.shape} done", file=sys.stderr, flush=True)
    pairs, sig_sim, stats = candidates_from_signatures(
        sig, rated_users(M), r, b, max_bucket=max_bucket, min_sig_sim=min_sig_sim,
        deadline=deadline, verbose=verbose, t0=t0,
    )
    stats["time_signatures"] = round(t_sig, 2)
    stats["time_total"] = round(time.perf_counter() - t0, 2)
    return pairs, sig_sim, stats


def candidates_from_signatures(
    sig,
    users,
    r,
    b=None,
    max_bucket=DEFAULT_MAX_BUCKET,
    min_sig_sim=DEFAULT_MIN_SIG_SIM,
    deadline=None,
    verbose=True,
    t0=None,
):
    h, n_users = sig.shape
    b = h // r if b is None else b
    if r < 1 or b < 1 or b * r > h:
        raise ValueError(f"need r, b >= 1 and b * r <= h, got h={h}, r={r}, b={b}")
    t_start = time.perf_counter()
    t0 = t_start if t0 is None else t0

    def log(msg):
        if verbose:
            print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    def out_of_time():
        return deadline is not None and time.perf_counter() > deadline

    sigT = np.ascontiguousarray(sig.T)
    if min_sig_sim is None:
        min_sig_sim = default_min_sig_sim(h)
    min_agree = math.ceil(min_sig_sim * h)
    stats = dict(
        users=len(users), h=h, r=r, b=b, max_bucket=max_bucket, min_sig_sim=round(min_sig_sim, 3),
        bands_done=0, pairs_generated=0, largest_bucket=0, big_buckets=0, big_buckets_done=0,
    )
    kept = []

    def keep(lo, hi):
        stats["pairs_generated"] += len(lo)
        if min_agree > 0:
            mask = agreeing_rows(sigT, lo, hi) >= min_agree
            lo, hi = lo[mask], hi[mask]
        kept.append(lo.astype(np.int64) * n_users + hi)

    deferred = []
    for band_idx in range(b):
        band = sig[band_idx * r:(band_idx + 1) * r]
        if len(users) < n_users:
            band = band[:, users]
        order, starts, sizes = band_buckets(band)
        members = users[order]
        small = sizes <= max_bucket
        for lo, hi in _small_bucket_pairs(members, starts[small], sizes[small]):
            keep(lo, hi)
            if out_of_time():
                break
        if out_of_time():
            log(f"time limit reached after {band_idx} of {b} bands")
            break
        deferred += [members[s:s + k].copy() for s, k in zip(starts[~small], sizes[~small])]
        if len(sizes):
            stats["largest_bucket"] = max(stats["largest_bucket"], int(sizes.max()))
        stats["bands_done"] += 1
    t_bands = time.perf_counter() - t_start
    log(f"{stats['bands_done']} bands done, {stats['pairs_generated']:,} pairs from buckets "
        f"<= {max_bucket}, {len(deferred)} big buckets postponed")

    # Big Buckets
    deferred.sort(key=len)
    stats["big_buckets"] = len(deferred)
    t_big0 = time.perf_counter()
    for bucket in deferred:
        if out_of_time():
            break
        for lo, hi in _big_bucket_pairs(bucket):
            keep(lo, hi)
            if out_of_time():
                break
        else:
            stats["big_buckets_done"] += 1
    t_big = time.perf_counter() - t_big0
    if deferred:
        log(f"big buckets: {stats['big_buckets_done']}/{len(deferred)} fully checked")

    keys = np.unique(np.concatenate(kept)) if kept else np.empty(0, dtype=np.int64)
    lo = (keys // n_users).astype(np.int32)
    hi = (keys % n_users).astype(np.int32)
    sig_sim = (agreeing_rows(sigT, lo, hi) / h).astype(np.float32)
    order = np.argsort(-sig_sim, kind="stable")
    pairs = np.column_stack([lo, hi])[order]
    sig_sim = sig_sim[order]

    stats.update(
        candidates=len(pairs),
        time_bands=round(t_bands, 2),
        time_big_buckets=round(t_big, 2),
        time_candidates=round(time.perf_counter() - t_start, 2),
    )
    log(f"{len(pairs):,} candidate pairs with sig_sim >= {min_sig_sim:.3f}")
    return pairs, sig_sim, stats


def default_min_sig_sim(h):
    return THRESHOLD - SIG_SIM_Z * math.sqrt(THRESHOLD * (1 - THRESHOLD) / h)


def s_curve(s, r, b):
    return 1.0 - (1.0 - np.asarray(s, dtype=float) ** r) ** b


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("--data", default="user_movie_rating.npy")
    p.add_argument("--h", type=int, default=DEFAULT_H)
    p.add_argument("--r", type=int, default=DEFAULT_R)
    p.add_argument("--b", type=int, default=None)
    p.add_argument("--max-bucket", type=int, default=DEFAULT_MAX_BUCKET)
    p.add_argument("--min-sig-sim", type=float, default=DEFAULT_MIN_SIG_SIM)
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    p.add_argument("--out", default="candidates.npy")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    M = load_user_movie_matrix(args.data)
    print(f"loaded {len(rated_users(M)):,} users x {M.shape[1] - 1:,} movies, {M.nnz:,} ratings "
          f"in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    pairs, sig_sim, stats = find_candidate_pairs(
        M, args.seed, h=args.h, r=args.r, b=args.b, max_bucket=args.max_bucket,
        min_sig_sim=args.min_sig_sim, time_limit=args.time_limit,
    )
    np.save(args.out, pairs)
    for k, v in stats.items():
        print(f"{k:>18}: {v}", file=sys.stderr)
    print(f"saved {len(pairs):,} pairs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
