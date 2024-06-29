"""Deprecated python fuctions for prior to porting to Rust."""

import heapq
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from libpysal import weights
from libpysal.cg import voronoi_frames
from statsmodels.distributions.empirical_distribution import ECDF

okabe = {
    "yellow": "#E69F00",
    "light_blue": "#56B4E9",
    "green": "#009E73",
    "amber": "#F5C710",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#999999",
    "black": "#000000",
}


def create_delaunay_graph(X: np.ndarray, clip: str = "convex hull"):
    """Create a Delaunay graph from a set of points.

    Parameters
    ----------
    X : np.ndarray
        An array of shape (n, 2) of points.
    clip : str, optional
        The method to use to clip the Voronoi diagram. By default, "convex hull".

    Returns
    -------
    nx.Graph
        A networkx graph representing the Delaunay graph.
    """
    cells, _ = voronoi_frames(X, clip=clip)
    delaunay = weights.Rook.from_dataframe(cells)
    return delaunay.to_networkx()


def plot_delaunay(df: str | pd.DataFrame, ax=None, with_graph=True):
    """Plot a Delaunay graph."""
    ax = ax or plt.gca()
    if isinstance(df, str):
        df = pd.read_parquet(f"../task2/parquet/{df}.parquet")
    points = df[["x", "y"]].to_numpy()
    graph = create_delaunay_graph(points)
    positions = dict(zip(graph.nodes, points))
    ax.scatter(df["x"], df["y"], c=df["label"].map(okabe))

    bounds = 160
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.set_aspect("equal", adjustable="box")

    if with_graph:
        nx.draw(
            graph,
            positions,
            node_size=1,
            node_color="k",
            edge_color="k",
            alpha=0.8,
            ax=ax,
        )


def dijkstra(
    graph: nx.Graph,
    start: int,
    points: np.ndarray,
    max_depth: float,
    prune_threshold: float,
):
    """Find the nodes within a certain distance of the start node.

    Parameters
    ----------
    graph : nx.Graph
        The graph to search.
    start : int
        The node to start the search from.
    points : np.ndarray
        The points corresponding to the nodes in the graph.
    max_depth : float
        The maximum depth to search.
    prune_threshold : float
        A threshold to prune the search.

    Returns
    -------
    set[int]
        The nodes within the specified distance of the start node.
    """
    visited = set()
    queue = [(0, start, 0)]  # (distance, node, depth)

    while queue:
        distance, node, depth = heapq.heappop(queue)

        if node not in visited:
            visited.add(node)

            if depth < max_depth:
                neighbors = graph[node]
                for neighbor in neighbors:
                    edge_weight = np.linalg.norm(points[node] - points[neighbor])
                    new_distance = distance + edge_weight

                    if new_distance <= prune_threshold and neighbor not in visited:
                        heapq.heappush(queue, (new_distance, neighbor, depth + 1))

    return visited


def bfs(graph: nx.Graph, start: int, max_depth: int):
    """Find the nodes within a certain distance of the start node.

    Parameters
    ----------
    graph : nx.Graph
        The graph to search.
    start : int
        The node to start the search from.
    max_depth : int
        The maximum depth to search.

    Returns
    -------
    set[int]
        The nodes within the specified distance of the start node.
    """
    visited = set()
    queue = deque([(start, 0)])

    while queue:
        node, depth = queue.popleft()

        if node not in visited:
            visited.add(node)

            if depth < max_depth:
                neighbors = graph[node]
                queue.extend((neighbor, depth + 1) for neighbor in neighbors)

    return visited


def find_all_self_edges(edges: np.ndarray, labels: np.ndarray, label: str):
    """Find all edges that are between the same label.

    Parameters
    ----------
    edges : np.ndarray
        An array of shape (n, 2) of edge indices.
    labels : np.ndarray
        An array of shape (n,) of labels.
    label : str
        The label to find edges for.

    Returns
    -------
    np.ndarray
        An array of shape (n, 2) of edge indices.
    """
    # get the edges that are between the same label
    mask = (labels[edges] == [label, label]).sum(axis=1) == 2
    # get the nodes that are on those edges
    return edges[mask]


def calculate_pairwise_distances(points: np.ndarray):
    """Get the average distance between points.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 2, 2) of points.

    Returns
    -------
    np.ndarray
        An array of shape (n,) of distances.
    """
    return np.linalg.norm(points[:, 0] - points[:, 1], axis=1)


@dataclass
class ConfusionResult:
    confusion: set[int]
    boundary_edges: np.ndarray


def find_label_confusion(
    points: np.ndarray,
    labels: np.ndarray,
    graph: nx.Graph,
    label: str,
    max_depth: int = 1,
    prune_factor: float = 2,
) -> ConfusionResult:
    """Find the confusion set for a label and its boundary edges.

    Boundary edges are edges from `label -> other`, where `other` is not
    in the confusion set. We store these edges so that we can use when
    weighting the neighborhood.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 2) of points.
    labels : np.ndarray
        An array of shape (n,) of labels.
    graph : nx.Graph
        The graph to search.
    label : str
        The label to find the confusion set for.
    max_depth : int, optional
        The maximum depth to search, by default 1.
    prune_factor : float, optional
        A factor to prune the search, by default 2. This is multiplied by the standard
        deviation of the distances between points on the same label and added to the
        mean to get a pruning threshold for the Dijkstra search. Points that are
        further than this threshold will not be searched.

    Returns
    -------
    ConfusionResult
        A tuple of the confusion set and boundary edges.
    """
    edges = np.array(graph.edges)
    self_edges = find_all_self_edges(edges, labels, label)
    self_distances = calculate_pairwise_distances(points[self_edges])
    threshold = self_distances.mean() + (prune_factor * self_distances.std())

    label_indices = np.where(labels == label)[0]

    confusion_with_edge_penalty = set.union(
        *(dijkstra(graph, idx, points, max_depth, threshold) for idx in label_indices)
    )

    confusion_without_edge_penalty = set.union(
        *(bfs(graph, idx, max_depth + 1) for idx in label_indices)
    )

    confusion_without_edge_penalty -= confusion_with_edge_penalty

    boundary_edges = []
    for u, v in graph.edges(confusion_with_edge_penalty):
        if v in confusion_without_edge_penalty:
            boundary_edges.append((u, v))

    return ConfusionResult(
        confusion=confusion_with_edge_penalty,
        boundary_edges=np.array(boundary_edges),
    )


def create_confusion_matrix(df: pd.DataFrame, infos: dict[str, ConfusionResult]):
    """Create a confusion matrix from the confusion results.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with a column named "label" that contains the labels.
    infos : dict[str, ConfusionResult]
        A dictionary of confusion results.

    Returns
    -------
    pd.DataFrame
        A dataframe with the confusion matrix.
    """
    label_categories = df["label"].cat.categories
    confusion = []
    for label in label_categories:
        inds = np.array(list(infos[label].confusion))
        s = df["label"].iloc[inds].value_counts()
        s.name = label
        confusion.append(s)

    df = pd.concat(confusion, axis=1).T
    return df.reindex(label_categories, columns=label_categories)


def map_range(distances: np.ndarray, min_val: float, max_val: float):
    """Map the distances to a range between 0 and 1."""
    out = (max_val - distances) / (max_val - min_val)
    out[distances > max_val] = 0
    out[distances < min_val] = 1
    return out


def find_label_neighborhood(
    points: np.ndarray,
    labels: np.ndarray,
    graph: nx.Graph,
    confusion_result: ConfusionResult,
    scale: tuple[float, float],
    depth_limit: int,
):
    """Find the neighborhood for a label."""
    boundary_edges = confusion_result.boundary_edges
    if len(boundary_edges) == 0:
        return []

    boundary_target_labels = labels[boundary_edges[:, 1]]
    boundary_edge_dist = calculate_pairwise_distances(points[boundary_edges])
    weights = map_range(boundary_edge_dist, scale[0], scale[1])
    # [(label, weight), ...]
    boundaries = list(zip(boundary_target_labels, weights, boundary_edge_dist))

    if depth_limit == 0:
        return boundaries

    for u, v in boundary_edges:
        visited = bfs(graph, v, depth_limit)
        edges = np.full((len(visited), 2), u)
        edges[:, 1] = list(visited)
        target_labels = labels[edges[:, 1]]
        edge_dist = calculate_pairwise_distances(points[edges])
        weights = map_range(edge_dist, scale[0], scale[1])
        boundaries.extend(zip(target_labels, weights, edge_dist))

    return boundaries


def find_neighborhoods(
    points: np.ndarray,
    labels: np.ndarray,
    graph: nx.Graph,
    max_depth: int = 1,
    confusion_results: dict[str, ConfusionResult] | None = None,
    scale: tuple[float | None, float | None] = (None, None),
    **kwargs,
):
    """Find the neighborhood for each label."""
    unique_labels = np.unique(labels)

    if confusion_results is None:
        confusion_results = {
            label: find_label_confusion(
                points=points,
                labels=labels,
                graph=graph,
                label=label,
                max_depth=1,
                **kwargs,
            )
            for label in unique_labels
        }
    boundary_edges = np.concatenate(
        [r.boundary_edges for r in confusion_results.values()]
    )
    # remove duplicates
    boundary_edges = np.unique(boundary_edges, axis=1)
    boundary_distances = calculate_pairwise_distances(points[boundary_edges])

    scale_min, scale_max = scale
    if scale_min is None:
        scale_min = boundary_distances.min()
    if scale_max is None:
        scale_max = boundary_distances.max()

    neighborhoods = {}
    for label, confusion_result in confusion_results.items():
        neighborhoods[label] = find_label_neighborhood(
            points=points,
            labels=labels,
            graph=graph,
            confusion_result=confusion_result,
            scale=(scale_min, scale_max),
            depth_limit=max_depth,
        )
    return neighborhoods


def neighborhood(df: pd.DataFrame, **kwargs):
    """Find the neighborhood for each label in the dataframe."""
    points = df[["x", "y"]].to_numpy()
    labels = df["label"].to_numpy()
    graph = create_delaunay_graph(points)
    return find_neighborhoods(points, labels, graph, **kwargs)


def confusion(df: pd.DataFrame, counts: bool = False, **kwargs):
    """Find the confusion set for each label in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with a column named "label" that contains the labels.
    counts : bool, optional
        If True, returns the full confusion matrix as a DataFrame, where
        each row is a count vector of the confusion set for a label.
        If False, returns a Series of confusion scores.
    kwargs : dict
        Additional keyword arguments for the confusion set.

    """
    points = df[["x", "y"]].to_numpy()
    labels = df["label"].to_numpy()
    graph = create_delaunay_graph(points)
    confusion_results = {
        label: find_label_confusion(
            points=points, labels=labels, graph=graph, label=label, **kwargs
        )
        for label in df["label"].cat.categories
    }
    mat = create_confusion_matrix(df, confusion_results)
    if counts:
        return mat
    normed = (mat / mat.sum(axis=1)).to_numpy()
    return pd.Series(1 - normed.diagonal(), index=mat.index, name="confusion")


def confusion_neighborhood(
    df: pd.DataFrame,
    confusion_kwargs: None | dict = None,
    neighborhood_kwargs: None | dict = None,
):
    """Find the confusion set and neighborhood for each label in the dataframe."""
    points = df[["x", "y"]].to_numpy()
    labels = df["label"].to_numpy()
    graph = create_delaunay_graph(points)

    confusion_results = {
        label: find_label_confusion(
            points=points, labels=labels, graph=graph, **(confusion_kwargs or {})
        )
        for label in df["label"].cat.categories
    }
    confusion_matrix = create_confusion_matrix(df, confusion_results)

    neighborhoods = find_neighborhoods(
        points=points,
        labels=labels,
        graph=graph,
        confusion_results=confusion_results,
        **(neighborhood_kwargs or {}),
    )

    return confusion_matrix, neighborhoods


def summarize_neighborhood(boundaries):
    """Summarize the neighborhood results into a DataFrame."""
    d = pd.DataFrame.from_records(boundaries, columns=["count", "weight", "dist"])
    df = pd.concat(
        [d["count"].value_counts(), d.groupby(["count"]).mean()["dist"]], axis=1
    )
    df.index.rename("label", inplace=True)
    return df


def process_neighborhood(nresult):
    """Process the neighborhood results into a DataFrame."""
    dfs = []
    for b in nresult.values():
        dfs.append(summarize_neighborhood(b))

    combined_stats = pd.concat(dfs)
    ecdf = ECDF(combined_stats["count"])
    dist_ecdf = ECDF(combined_stats["dist"])

    labels = list(nresult.keys())

    freqs = []
    for label, stats in zip(labels, dfs):
        row = pd.Series(np.zeros(len(labels)), index=labels, name=label)
        counts = ecdf(stats["count"])
        dist = dist_ecdf(stats["dist"])
        row.loc[stats.index] = counts * (1 - dist)
        freqs.append(row)

    df = pd.concat(freqs, axis=1).T
    return df.reindex(labels, columns=labels)


def count_bits(uint: np.ndarray):
    """Count the number of bits set to 1 in each uint8 value.

    Parameters
    ----------
    uint : np.ndarray
        Array of uint8 values.

    Returns
    -------
    np.ndarray
        Array of counts of bits set to 1.
    """
    assert uint.dtype == np.uint8, "must be uint8"
    bits = np.unpackbits(uint[:, :, np.newaxis].view("uint8"), axis=2)
    return bits.sum(axis=2)


def jaccard_distance(v1: np.ndarray, v2: np.ndarray):
    """Compute the Jaccard distance between two uint8 arrays using bit operations.

    Parameters
    ----------
    v1 : np.ndarray
        Array of uint8 values.
    v2 : np.ndarray
        Array of uint8 values.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Array of Jaccard distances.
    """
    assert v1.dtype == np.uint8 and v2.dtype == np.uint8, "must be uint8"

    intersection = count_bits(np.bitwise_and(v1, v2))
    union = count_bits(np.bitwise_or(v1, v2))

    # empty sets result in a divide by zero.
    # we consider the similarity of {} and {} to be 1.
    intersection[union == 0] = 1
    union[union == 0] = 1

    return 1 - (intersection / union)


def hamming_distance(v1: np.ndarray, v2: np.ndarray):
    """Compute the Hamming distance between two uint8 arrays using bit operations.

    Parameters
    ----------
    v1 : np.ndarray
        Array of uint8 values.
    v2 : np.ndarray
        Array of uint8 values.

    Returns
    -------
    np.ndarray
        Array of Hamming distances.
    """
    assert v1.dtype == np.uint8 and v2.dtype == np.uint8, "must be uint8"
    return count_bits(np.bitwise_xor(v1, v2))
