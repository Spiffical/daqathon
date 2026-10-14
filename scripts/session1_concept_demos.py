"""Small synthetic demos that explain model ideas before the real-data sections.

These figures use made-up data on purpose: they show one idea at a time, and
they work the same way whichever dataset preset is active.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _nearest_centroid(points: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's nearest-centroid index and its distance to that centroid."""
    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return distances.argmin(axis=1), distances.min(axis=1)


def plot_kmeans_iterations_demo(n_clusters: int = 3, seed: int = 21) -> plt.Figure:
    """Show the k-means loop on a toy 2D dataset, one step per panel.

    Panels:

    1. Start: place ``n_clusters`` centroids (here, deliberately in a poor spot).
    2. Assign: every point joins its nearest centroid.
    3. Update: every centroid moves to the mean of the points assigned to it.
    4. Converged: steps 2 and 3 repeat until no point changes cluster. One
       point's ``distance_to_centroid`` is drawn as a dashed line.
    """
    rng = np.random.default_rng(seed)
    blob_centres = np.array([[-2.0, -1.2], [2.0, -0.8], [0.2, 2.0]])[:n_clusters]
    points = np.vstack([centre + rng.normal(scale=0.55, size=(60, 2)) for centre in blob_centres])

    # Start every centroid near the bottom-left so the later panels show movement.
    left_points = points[np.argsort(points[:, 0])[: max(n_clusters * 4, 12)]]
    centroids = left_points[rng.choice(len(left_points), size=n_clusters, replace=False)].copy()
    start_centroids = centroids.copy()

    first_assignment, _ = _nearest_centroid(points, centroids)
    first_update = np.array(
        [points[first_assignment == k].mean(axis=0) if np.any(first_assignment == k) else centroids[k] for k in range(n_clusters)]
    )

    assignment = first_assignment
    centroids = first_update
    iterations = 1
    for iterations in range(2, 51):
        new_assignment, _ = _nearest_centroid(points, centroids)
        centroids = np.array(
            [points[new_assignment == k].mean(axis=0) if np.any(new_assignment == k) else centroids[k] for k in range(n_clusters)]
        )
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
    final_assignment, final_distance = _nearest_centroid(points, centroids)

    colours = plt.cm.tab10(np.arange(n_clusters))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)

    axes[0].scatter(points[:, 0], points[:, 1], s=14, color="#94a3b8")
    axes[0].scatter(start_centroids[:, 0], start_centroids[:, 1], s=260, marker="*", color=colours, edgecolors="black")
    axes[0].set_title(f"1. Start: place k={n_clusters} centroids")

    axes[1].scatter(points[:, 0], points[:, 1], s=14, color=colours[first_assignment])
    axes[1].scatter(start_centroids[:, 0], start_centroids[:, 1], s=260, marker="*", color=colours, edgecolors="black")
    axes[1].set_title("2. Assign each point to\nits nearest centroid")

    axes[2].scatter(points[:, 0], points[:, 1], s=14, color=colours[first_assignment], alpha=0.5)
    for k in range(n_clusters):
        axes[2].annotate(
            "",
            xy=first_update[k],
            xytext=start_centroids[k],
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.4},
        )
    axes[2].scatter(first_update[:, 0], first_update[:, 1], s=260, marker="*", color=colours, edgecolors="black")
    axes[2].set_title("3. Move each centroid to the\nmean of its assigned points")

    axes[3].scatter(points[:, 0], points[:, 1], s=14, color=colours[final_assignment])
    axes[3].scatter(centroids[:, 0], centroids[:, 1], s=260, marker="*", color=colours, edgecolors="black")
    far_point = int(final_distance.argmax())
    far_centroid = centroids[final_assignment[far_point]]
    axes[3].plot(
        [points[far_point, 0], far_centroid[0]],
        [points[far_point, 1], far_centroid[1]],
        linestyle="--",
        color="black",
        linewidth=1.2,
    )
    axes[3].annotate(
        f"distance_to_centroid = {final_distance[far_point]:.2f}",
        xy=(points[far_point] + far_centroid) / 2,
        xytext=(12, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axes[3].set_title(f"4. Repeat 2–3 until nothing changes\n(converged after {iterations} rounds)")

    for axis in axes:
        axis.set_xlabel("feature 1 (standardised)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("feature 2 (standardised)")
    fig.suptitle("How k-means finds clusters (toy 2D example)", y=1.04)
    fig.tight_layout()
    return fig


def make_conductivity_plug_demo_signal(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic 1 Hz conductivity record with a spike and a plug-like drop."""
    rng = np.random.default_rng(seed)
    seconds = np.arange(600)
    signal = 3.20 + 0.01 * np.sin(seconds / 60) + rng.normal(scale=0.006, size=seconds.size)
    signal[120] += 0.12  # one-sample spike
    signal[260:400] -= 0.55  # sudden drop when the conductivity cell is blocked
    recovery = np.linspace(0.55, 0.0, 30)
    signal[400:430] -= recovery  # gradual recovery once the blockage clears
    return seconds, signal


def plot_convolution_filter_demo(filter_weights: list[float] | np.ndarray) -> plt.Figure:
    """Slide one filter along a synthetic conductivity record and plot the response.

    The response at each time step is the dot product between the filter and
    the stretch of signal under it. This is what one channel of a ``Conv1d``
    layer computes (PyTorch slides the filter without flipping it).
    """
    weights = np.asarray(filter_weights, dtype=float)
    if weights.ndim != 1 or weights.size < 2:
        raise ValueError("filter_weights must be a list of at least two numbers.")

    seconds, signal = make_conductivity_plug_demo_signal()
    # Remove the typical level first so the response shows shape, not offset.
    centred_signal = signal - np.median(signal)
    response = np.correlate(centred_signal, weights, mode="same")

    fig = plt.figure(figsize=(14, 7.6))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.6, 1.6], hspace=0.6)

    filter_axis = fig.add_subplot(grid[0, 0])
    filter_axis.bar(np.arange(weights.size), weights, color=np.where(weights >= 0, "#2563eb", "#dc2626"))
    filter_axis.axhline(0, color="black", linewidth=0.8)
    filter_axis.set_title("1. The filter (its weights)")
    filter_axis.set_xlabel("position in filter")
    filter_axis.set_xticks(np.arange(weights.size))

    signal_axis = fig.add_subplot(grid[1, :])
    signal_axis.plot(seconds, signal, color="#0f172a", linewidth=1.2)
    signal_axis.set_title("2. The input: a synthetic 1 Hz conductivity record with a one-point spike (t=120 s) and a plug-like drop (t=260 s)")
    signal_axis.set_ylabel("Conductivity (S/m)")
    signal_axis.grid(alpha=0.25)

    response_axis = fig.add_subplot(grid[2, :], sharex=signal_axis)
    response_axis.plot(seconds, response, color="#7c3aed", linewidth=1.4)
    response_axis.axhline(0, color="black", linewidth=0.8)
    strongest = int(np.abs(response).argmax())
    response_axis.scatter([seconds[strongest]], [response[strongest]], color="#7c3aed", s=50, zorder=3)
    response_axis.annotate(
        f"strongest response at t={seconds[strongest]} s",
        xy=(seconds[strongest], response[strongest]),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
    )
    response_axis.set_title("3. The output: the filter's response at every time step (large magnitude = pattern found)")
    response_axis.set_xlabel("time (s)")
    response_axis.set_ylabel("response")
    response_axis.grid(alpha=0.25)
    return fig
