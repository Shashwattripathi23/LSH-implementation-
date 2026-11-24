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
    """Binary CSR matrix with row index = user_id and column index = movie_id."""
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
    """(h, n_users) matrix with sig[i, u] = min of perm_i[m] over the movies m rated by u.

    perm_i = rng.permutation(n_movies) gives every movie a rank. Movies are visited in rank
    order and each user gets the rank of the first visited movie they rated; the scan stops
    as soon as every user has a value. Users without ratings keep the value n_movies.
    """
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
    """Group the identical columns of an (r, n) band.

    Returns (order, starts, sizes): bucket j holds columns order[starts[j]:starts[j] + sizes[j]].
    Only buckets with at least two columns are returned.
    """
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
    """Number of minhashes on which users lo[k] and hi[k] agree (sigT is the (n_users, h) signature)."""
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
    """Minhash + LSH banding on a user x movie sparse matrix.

    Returns (pairs, sig_sim, stats):
      pairs    int32 (n, 2) row indices of M, pairs[:, 0] < pairs[:, 1], unique,
               sorted by sig_sim from high to low
      sig_sim  float32 (n,) fraction of the h minhashes on which the pair agrees
      stats    dict of counts and timings
    time_limit (seconds) caps this whole call; postponed big buckets only get the time that is left.
    """
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
    """Banding, buckets and candidate pairs from a precomputed (h, n_users) signature matrix.

    users lists the columns that take part (users with at least one rating).
    deadline is an absolute time.perf_counter() value after which no more pairs are generated.
    """
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
    """Lowest signature agreement kept: THRESHOLD minus SIG_SIM_Z standard errors of an h-row estimate."""
    return THRESHOLD - SIG_SIM_Z * math.sqrt(THRESHOLD * (1 - THRESHOLD) / h)


def s_curve(s, r, b):
    """Probability that a pair with Jaccard s lands in the same bucket in at least one of b bands of r rows."""
    return 1.0 - (1.0 - np.asarray(s, dtype=float) ** r) ** b


def main(argv=None):
    p = argparse.ArgumentParser(description="Minhash + LSH banding: candidate pairs of similar Netflix users.")
    p.add_argument("seed", type=int, help="random seed")
    p.add_argument("--data", default="user_movie_rating.npy", help="ratings .npy (user_id, movie_id, rating)")
    p.add_argument("--h", type=int, default=DEFAULT_H, help="signature length (number of permutations)")
    p.add_argument("--r", type=int, default=DEFAULT_R, help="rows per band")
    p.add_argument("--b", type=int, default=None, help="number of bands (default h // r)")
    p.add_argument("--max-bucket", type=int, default=DEFAULT_MAX_BUCKET,
                   help="buckets with more users than this are postponed to the end")
    p.add_argument("--min-sig-sim", type=float, default=DEFAULT_MIN_SIG_SIM,
                   help="drop pairs whose signatures agree on fewer than this fraction of rows "
                        "(default: 0.5 minus 3.5 standard errors, 0.40 at h=300; 0 keeps all)")
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                   help="seconds allowed for minhashing + banding")
    p.add_argument("--out", default="candidates.npy", help="where to save the (n, 2) candidate array")
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
