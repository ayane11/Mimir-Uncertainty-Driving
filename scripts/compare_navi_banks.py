#!/usr/bin/env python3
"""Compare two navi bank npy files."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def _load_bank(path: Path):
    import numpy as np

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict npy: {path}")
    return data


def _to_points(value, token: str):
    import numpy as np

    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 1:
        if points.shape[0] < 2:
            raise ValueError(f"Expected endpoint with at least 2 dims for token {token}, got shape {points.shape}")
        return points[:2].reshape(1, 2)
    if points.shape[-1] < 2:
        raise ValueError(f"Expected endpoints with last dim >= 2 for token {token}, got shape {points.shape}")
    return points.reshape(-1, points.shape[-1])[:, :2]


def _nearest_pair(points_a, points_b):
    import numpy as np

    diff = points_b[None, :, :] - points_a[:, None, :]
    distances = np.linalg.norm(diff, axis=-1)
    a_idx, b_idx = np.unravel_index(int(distances.argmin()), distances.shape)
    return a_idx, b_idx, points_a[a_idx], points_b[b_idx], diff[a_idx, b_idx], float(distances[a_idx, b_idx])


def _nearest_internal_pair(points):
    import numpy as np

    if points.shape[0] < 2:
        return None
    diff = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(distances, np.inf)
    a_idx, b_idx = np.unravel_index(int(distances.argmin()), distances.shape)
    return a_idx, b_idx, points[a_idx], points[b_idx], diff[a_idx, b_idx], float(distances[a_idx, b_idx])


def _percentile(values, q: float):
    import numpy as np

    return float(np.percentile(values, q))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare navi bank npy files.")
    parser.add_argument("bank_a", type=Path, help="First navi bank npy.")
    parser.add_argument("bank_b", type=Path, nargs="?", help="Second navi bank npy. Omit it to compare endpoints within bank_a.")
    parser.add_argument("--top", type=int, default=20, help="Number of samples to print.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV with per-token distances.")
    args = parser.parse_args()

    import numpy as np

    bank_a = _load_bank(args.bank_a)
    if args.bank_b is None:
        rows = []
        endpoint_counts = Counter()
        for token, value in sorted(bank_a.items()):
            points = _to_points(value, token)
            endpoint_counts[points.shape[0]] += 1
            nearest = _nearest_internal_pair(points)
            if nearest is None:
                continue
            a_idx, b_idx, point_a, point_b, diff, dist = nearest
            rows.append((token, points.shape[0], a_idx, b_idx, point_a, point_b, diff, dist))

        print(f"bank entries: {len(bank_a)}")
        print(f"endpoint counts: {dict(sorted(endpoint_counts.items()))}")
        print(f"tokens with >=2 endpoints: {len(rows)}")
        print("distance mode: nearest pair within endpoints per token")

        if len(rows) == 0:
            return 0

        distances = np.asarray([row[-1] for row in rows], dtype=np.float64)
        abs_dx = np.asarray([abs(row[6][0]) for row in rows], dtype=np.float64)
        abs_dy = np.asarray([abs(row[6][1]) for row in rows], dtype=np.float64)

        print("distance summary:")
        print(f"  mean={distances.mean():.6f}, std={distances.std():.6f}, min={distances.min():.6f}, max={distances.max():.6f}")
        print(
            "  "
            f"p01={_percentile(distances, 1):.6f}, "
            f"p05={_percentile(distances, 5):.6f}, "
            f"p50={_percentile(distances, 50):.6f}, "
            f"p95={_percentile(distances, 95):.6f}"
        )
        print(f"  mean_abs_dx={abs_dx.mean():.6f}, mean_abs_dy={abs_dy.mean():.6f}")
        print("thresholds:")
        for threshold in (1e-6, 0.1, 0.5, 1.0, 2.0, 5.0):
            count = int((distances <= threshold).sum())
            print(f"  <= {threshold:g}m: {count} ({count / len(rows) * 100:.2f}%)")

        print(f"top {min(args.top, len(rows))} closest endpoint pairs:")
        for token, count, a_idx, b_idx, point_a, point_b, diff, dist in sorted(rows, key=lambda row: row[-1])[: args.top]:
            print(
                f"  {token}: dist={dist:.6f}, "
                f"idx={a_idx}/{count} vs {b_idx}/{count}, "
                f"a=({point_a[0]:.6f}, {point_a[1]:.6f}), "
                f"b=({point_b[0]:.6f}, {point_b[1]:.6f}), "
                f"diff=({diff[0]:.6f}, {diff[1]:.6f})"
            )

        if args.csv is not None:
            with args.csv.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["token", "count", "a_idx", "b_idx", "a_x", "a_y", "b_x", "b_y", "dx", "dy", "l2"])
                for token, count, a_idx, b_idx, point_a, point_b, diff, dist in rows:
                    writer.writerow([token, count, a_idx, b_idx, point_a[0], point_a[1], point_b[0], point_b[1], diff[0], diff[1], dist])
            print(f"saved csv: {args.csv}")
        return 0

    bank_b = _load_bank(args.bank_b)
    keys_a = set(bank_a)
    keys_b = set(bank_b)
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    rows = []
    for token in common:
        points_a = _to_points(bank_a[token], token)
        points_b = _to_points(bank_b[token], token)
        a_idx, b_idx, point_a, point_b, diff, dist = _nearest_pair(points_a, points_b)
        rows.append((token, points_a.shape[0], points_b.shape[0], a_idx, b_idx, point_a, point_b, diff, dist))

    distances = np.asarray([row[-1] for row in rows], dtype=np.float64)
    abs_dx = np.asarray([abs(row[7][0]) for row in rows], dtype=np.float64)
    abs_dy = np.asarray([abs(row[7][1]) for row in rows], dtype=np.float64)
    endpoints_a = Counter(row[1] for row in rows)
    endpoints_b = Counter(row[2] for row in rows)

    print(f"bank_a entries: {len(bank_a)}")
    print(f"bank_b entries: {len(bank_b)}")
    print(f"common entries: {len(common)}")
    print(f"only in bank_a: {len(only_a)}")
    print(f"only in bank_b: {len(only_b)}")

    if len(rows) == 0:
        return 0

    print(f"bank_a endpoint counts: {dict(sorted(endpoints_a.items()))}")
    print(f"bank_b endpoint counts: {dict(sorted(endpoints_b.items()))}")
    print("distance mode: nearest pair over all endpoints per token")
    print("distance summary:")
    print(f"  mean={distances.mean():.6f}, std={distances.std():.6f}, max={distances.max():.6f}")
    print(
        "  "
        f"p50={_percentile(distances, 50):.6f}, "
        f"p90={_percentile(distances, 90):.6f}, "
        f"p95={_percentile(distances, 95):.6f}, "
        f"p99={_percentile(distances, 99):.6f}"
    )
    print(f"  mean_abs_dx={abs_dx.mean():.6f}, mean_abs_dy={abs_dy.mean():.6f}")
    print("thresholds:")
    for threshold in (1e-6, 0.1, 0.5, 1.0, 2.0, 5.0):
        count = int((distances <= threshold).sum())
        print(f"  <= {threshold:g}m: {count} ({count / len(rows) * 100:.2f}%)")

    print(f"top {min(args.top, len(rows))} largest differences:")
    for token, a_count, b_count, a_idx, b_idx, point_a, point_b, diff, dist in sorted(rows, key=lambda row: row[-1], reverse=True)[
        : args.top
    ]:
        print(
            f"  {token}: dist={dist:.6f}, "
            f"a_idx={a_idx}/{a_count}, b_idx={b_idx}/{b_count}, "
            f"a=({point_a[0]:.6f}, {point_a[1]:.6f}), "
            f"b=({point_b[0]:.6f}, {point_b[1]:.6f}), "
            f"diff=({diff[0]:.6f}, {diff[1]:.6f})"
        )

    if args.csv is not None:
        with args.csv.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["token", "a_count", "b_count", "a_idx", "b_idx", "a_x", "a_y", "b_x", "b_y", "dx", "dy", "l2"])
            for token, a_count, b_count, a_idx, b_idx, point_a, point_b, diff, dist in rows:
                writer.writerow(
                    [token, a_count, b_count, a_idx, b_idx, point_a[0], point_a[1], point_b[0], point_b[1], diff[0], diff[1], dist]
                )
        print(f"saved csv: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
