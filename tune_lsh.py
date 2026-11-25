import argparse
import csv
import sys
import time

import numpy as np

from lsh_candidates import (
    DEFAULT_MAX_BUCKET,
    DEFAULT_MIN_SIG_SIM,
    DEFAULT_TIME_LIMIT,
    THRESHOLD,
    candidates_from_signatures,
    load_user_movie_matrix,
    minhash_signatures,
    rated_users,
    s_curve,
)


def exact_jaccard(M, lo, hi, chunk=20_000):
    """Exact Jaccard similarity of the rated-movie sets of users lo[k] and hi[k]."""
    M = M.tocsr()
    sizes = np.diff(M.indptr)
    out = np.empty(len(lo), dtype=np.float64)
    for s in range(0, len(lo), chunk):
        a, c = lo[s:s + chunk], hi[s:s + chunk]
        inter = np.asarray(M[a].multiply(M[c]).sum(axis=1)).ravel()
        out[s:s + chunk] = inter / (sizes[a] + sizes[c] - inter)
    return out


def sweep(M, seeds, hs, rs, max_bucket, min_sig_sim, time_limit):
    users = rated_users(M)
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        sig_full = minhash_signatures(M, max(hs), np.random.default_rng(seed))
        t_sig_full = time.perf_counter() - t0
        for h in hs:
            t_sig = t_sig_full * h / max(hs)
            for r in rs:
                if r > h:
                    continue
                deadline = time.perf_counter() + time_limit - t_sig
                pairs, _, st = candidates_from_signatures(
                    sig_full[:h], users, r, max_bucket=max_bucket, min_sig_sim=min_sig_sim,
                    deadline=deadline, verbose=False,
                )
                t1 = time.perf_counter()
                jac = exact_jaccard(M, pairs[:, 0], pairs[:, 1])
                t_check = time.perf_counter() - t1
                row = dict(
                    seed=seed, h=h, r=r, b=st["b"],
                    p_at_0_5=round(float(s_curve(THRESHOLD, r, st["b"])), 3),
                    pairs_generated=st["pairs_generated"], candidates=st["candidates"],
                    similar_pairs=int((jac > THRESHOLD).sum()),
                    bands_done=st["bands_done"], largest_bucket=st["largest_bucket"],
                    big_buckets=st["big_buckets"],
                    big_buckets_done=st["big_buckets_done"],
                    time_signatures=round(t_sig, 1), time_candidates=st["time_candidates"],
                    time_check=round(t_check, 1),
                )
                rows.append(row)
                print(", ".join(f"{k}={v}" for k, v in row.items()), file=sys.stderr, flush=True)
    return rows


def summarize(rows):
    keys = sorted({(r["h"], r["r"]) for r in rows})
    print(f"\n{'h':>4} {'r':>3} {'b':>3} {'P(0.5)':>7} {'similar':>8} {'min':>5} {'max':>5} "
          f"{'cands':>10} {'generated':>13} {'secs':>6}")
    for h, r in keys:
        g = [x for x in rows if x["h"] == h and x["r"] == r]
        found = [x["similar_pairs"] for x in g]
        secs = np.mean([x["time_signatures"] + x["time_candidates"] + x["time_check"] for x in g])
        print(f"{h:>4} {r:>3} {g[0]['b']:>3} {g[0]['p_at_0_5']:>7} {np.mean(found):>8.1f} "
              f"{min(found):>5} {max(found):>5} {np.mean([x['candidates'] for x in g]):>10,.0f} "
              f"{np.mean([x['pairs_generated'] for x in g]):>13,.0f} {secs:>6.0f}")


def plot_s_curves(configs, path):
    """S-curves for up to three (h, r) configurations, with the 0.5 threshold marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    styles = ["-", "--", ":"]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    s = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for (h, r), color, ls in zip(configs[:3], colors, styles):
        b = h // r
        p05 = float(s_curve(THRESHOLD, r, b))
        ax.plot(s, s_curve(s, r, b), color=color, lw=2, ls=ls, label=f"h={h}, r={r}, b={b}:  P(0.5) = {p05:.2f}")
        ax.plot([THRESHOLD], [p05], "o", ms=6, color=color, mec="#fcfcfb", mew=2, zorder=3)
    ax.axvline(THRESHOLD, color=muted, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(THRESHOLD - 0.008, 0.54, "Jaccard threshold 0.5", color=muted, fontsize=9, rotation=90, ha="right", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Jaccard similarity of a pair", color=ink)
    ax.set_ylabel("P(pair becomes a candidate)", color=ink)
    ax.grid(True, color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sweep signature length h and rows per band r on the real data.")
    p.add_argument("--s-curves", nargs="+", metavar="H:R",
                   help="only plot S-curves for these configs (e.g. 300:5 150:6) to --plot, no data needed")
    p.add_argument("--plot", default="s_curves.png")
    p.add_argument("--data", default="user_movie_rating.npy")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--h", type=int, nargs="+", default=[300, 450, 600])
    p.add_argument("--r", type=int, nargs="+", default=[5, 6])
    p.add_argument("--max-bucket", type=int, default=DEFAULT_MAX_BUCKET)
    p.add_argument("--min-sig-sim", type=float, default=DEFAULT_MIN_SIG_SIM)
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                   help="seconds allowed per run for minhashing + banding, as in lsh_candidates.py")
    p.add_argument("--csv", default="tuning_results.csv")
    args = p.parse_args(argv)

    if args.s_curves:
        plot_s_curves([tuple(map(int, c.split(":"))) for c in args.s_curves], args.plot)
        print(f"saved {args.plot}", file=sys.stderr)
        return

    M = load_user_movie_matrix(args.data)
    rows = sweep(M, args.seeds, args.h, args.r, args.max_bucket, args.min_sig_sim, args.time_limit)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nper-run results saved to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
