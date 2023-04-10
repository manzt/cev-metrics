use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashSet, VecDeque};

use delaunator::{triangulate, Point};
use numpy::ndarray::{Array, Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use petgraph::data::{Element, FromElements};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, VisitMap};

mod step_function;

fn euclidean_distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

/// A set of labels for a graph
/// Indices of the labels correspond to the indices of the nodes in the graph
/// The labels are represented as integers between 0 and n_categories - 1.
#[derive(Debug)]
struct Labels<'a> {
    codes: &'a [i16],
    n_categories: usize,
}

#[derive(Debug)]
struct ConfusionResult<'a> {
    set: HashSet<NodeIndex>,
    boundaries: HashSet<EdgeIndex>,
    labels: &'a Labels<'a>,
}

impl<'a> ConfusionResult<'a> {
    fn contains(&self, node: &NodeIndex) -> bool {
        self.set.contains(&node)
    }
}

trait ConfusionMatrix {
    fn counts(&self) -> Array2<u64>;
}

impl<'a> ConfusionMatrix for Vec<ConfusionResult<'a>> {
    fn counts(&self) -> Array2<u64> {
        let n = self[0].labels.n_categories;
        let codes = self[0].labels.codes;
        let mut data = Array::from_elem((n, n), 0u64);
        for (i, result) in self.iter().enumerate() {
            for node in result.set.iter() {
                let code = codes[node.index()];
                data[[i, code as usize]] += 1;
            }
        }
        data
    }
}

#[derive(Debug)]
struct NeighborhoodResult<'a> {
    distances: Vec<(usize, f64)>,
    labels: &'a Labels<'a>,
}

impl<'a> NeighborhoodResult<'a> {
    fn summarize(&self) -> Vec<Option<(i32, f64)>> {
        let mut data = vec![None; self.labels.n_categories];
        for (label, distance) in &self.distances {
            data[*label] = match data[*label] {
                None => Some((1, *distance)),
                Some(x) => Some((x.0 + 1, x.1 + *distance)),
            }
        }
        for entry in data.iter_mut() {
            if let Some((count, total)) = entry {
                *total = *total / *count as f64;
            }
        }
        data
    }
}

trait NeighborhoodSummary {
    fn scores(&self) -> Array2<f64>;
}

impl<'a> NeighborhoodSummary for Vec<NeighborhoodResult<'a>> {
    fn scores(&self) -> Array2<f64> {
        let summaries: Vec<_> = self.iter().map(|n| n.summarize()).collect();

        let counts: Vec<_> = summaries
            .iter()
            .flat_map(|summary| {
                summary
                    .iter()
                    .filter_map(|entry| entry.and_then(|x| Some(x.0 as f64)))
            })
            .collect();

        let distances: Vec<_> = summaries
            .iter()
            .flat_map(|summary| {
                summary
                    .iter()
                    .filter_map(|entry| entry.and_then(|x| Some(x.1)))
            })
            .collect();

        let ecdf_counts = step_function::ECDF::new(&counts, step_function::Side::Left);
        let ecdf_dist = step_function::ECDF::new(&distances, step_function::Side::Left);

        let n = self[0].labels.n_categories;
        let mut data = Array::from_elem((n, n), 0f64);

        for i in 0..n {
            for j in 0..n {
                if let Some((count, dist)) = summaries[i][j] {
                    let count_score = ecdf_counts.evaluate(count as f64);
                    let dist_score = ecdf_dist.evaluate(dist);
                    data[[i, j]] = count_score * (1.0 - dist_score);
                }
            }
        }

        data
    }
}

impl<'a> Labels<'a> {
    fn from_codes(codes: &'a [i16]) -> Self {
        let max = *codes.iter().max().unwrap();
        Self {
            codes,
            n_categories: (max + 1) as usize,
        }
    }

    fn confusion_threshold(&self, graph: &Graph) -> Vec<f64> {
        graph
            .graph
            .raw_edges()
            .iter()
            .fold(vec![vec![]; self.n_categories], |mut data, edge| {
                let source = edge.source().index();
                let target = edge.target().index();
                if self.codes[source] == self.codes[target] {
                    let code = self.codes[source] as usize;
                    data[code].push(edge.weight);
                }
                data
            })
            .iter()
            .map(|distances| {
                let n = distances.len() as f64;
                let mean = distances.iter().sum::<f64>() / n;
                let std = distances.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
                mean + 3.0 * std
            })
            .collect()
    }

    fn confusion_for_label(
        &self,
        graph: &Graph,
        label: i16,
        threshold: Option<f64>,
    ) -> ConfusionResult {
        let nodes: Vec<_> = graph
            .graph
            .node_indices()
            .filter(|node| self.codes[node.index()] == label)
            .collect();

        let (visited_with_threshold, visited_without_threshold) = nodes
            .par_iter()
            .map(|node| (graph.bfs(*node, 1, threshold), graph.bfs(*node, 2, None)))
            .reduce(
                || (HashSet::new(), HashSet::new()),
                |mut acc, (inner, outer)| {
                    acc.0.extend(inner);
                    acc.1.extend(outer);
                    acc
                },
            );

        // TODO, avoid second pass? Can we save edges found in bfs?
        let mut boundary_edges = HashSet::new();

        for source in visited_with_threshold {
            for edge in graph.graph.edges(source) {
                if visited_without_threshold.contains(&edge.target()) {
                    boundary_edges.insert(edge.id());
                }
            }
        }

        ConfusionResult {
            set: visited_without_threshold,
            boundaries: boundary_edges,
            labels: &self,
        }
    }

    fn confusion(&self, graph: &Graph) -> Vec<ConfusionResult> {
        (0..self.n_categories)
            .zip(self.confusion_threshold(graph))
            .map(|(label, threshold)| {
                self.confusion_for_label(graph, label as i16, Some(threshold))
            })
            .collect()
    }

    fn neighborhood_for_label(
        &self,
        graph: &Graph,
        confusion_result: &ConfusionResult,
        max_depth: usize,
    ) -> NeighborhoodResult {
        let boundary_distances = confusion_result.boundaries.iter().filter_map(|edge_index| {
            let edge = &graph.graph.raw_edges()[edge_index.index()];
            let target_label = self.codes[edge.target().index()];
            if confusion_result.contains(&edge.target()) {
                None
            } else {
                Some((target_label as usize, edge.weight))
            }
        });

        let visited: Vec<_> = confusion_result
            .boundaries
            .par_iter()
            .map(|edge_index| {
                let edge = &graph.graph.raw_edges()[edge_index.index()];
                (edge, graph.bfs(edge.target(), max_depth, None))
            })
            .collect();

        let connections = visited.iter().flat_map(|(edge, targets)| {
            let source_point = &graph.points[edge.source().index()];
            targets
                .iter()
                .filter(|target| !confusion_result.contains(&target))
                .map(|target| {
                    let target_label = self.codes[target.index()] as usize;
                    let target_point = &graph.points[target.index()];
                    (target_label, euclidean_distance(source_point, target_point))
                })
        });

        NeighborhoodResult {
            distances: boundary_distances.chain(connections).collect(),
            labels: &self,
        }
    }

    fn neighborhood(
        &self,
        graph: &Graph,
        counfusion_results: &Vec<ConfusionResult>,
        max_depth: usize,
    ) -> Vec<NeighborhoodResult> {
        counfusion_results
            .iter()
            .map(|c| self.neighborhood_for_label(graph, c, max_depth))
            .collect()
    }
}

#[pyclass(unsendable)]
#[derive(Debug)]
struct Graph {
    graph: UnGraph<usize, f64>,
    points: Vec<Point>,
}

impl From<&Vec<Point>> for Graph {
    fn from(points: &Vec<Point>) -> Self {
        let mut graph = UnGraph::<_, _>::from_elements(
            std::iter::repeat(Element::Node { weight: 0 }).take(points.len()),
        );
        for triangle in triangulate(&points).triangles.chunks(3) {
            let (a, b, c) = (triangle[0], triangle[1], triangle[2]);
            // `update_edge` avoids adding duplicate edges
            graph.update_edge(
                NodeIndex::new(a),
                NodeIndex::new(b),
                euclidean_distance(&points[a], &points[b]),
            );
            graph.update_edge(
                NodeIndex::new(b),
                NodeIndex::new(c),
                euclidean_distance(&points[b], &points[c]),
            );
            graph.update_edge(
                NodeIndex::new(c),
                NodeIndex::new(a),
                euclidean_distance(&points[c], &points[a]),
            );
        }
        Self {
            graph,
            points: points.clone(),
        }
    }
}

impl From<&PyReadonlyArray2<'_, f64>> for Graph {
    fn from(points: &PyReadonlyArray2<'_, f64>) -> Self {
        let points: Vec<_> = points
            .as_array()
            .lanes(Axis(1))
            .into_iter()
            .map(|x| Point { x: x[0], y: x[1] })
            .collect();
        Self::from(&points)
    }
}

impl Graph {
    fn bfs(
        &self,
        start: NodeIndex,
        max_depth: usize,
        threshold: Option<f64>,
    ) -> HashSet<NodeIndex> {
        let mut discovered = HashSet::new();
        discovered.visit(start);
        let mut stack = VecDeque::new();
        stack.push_front((start, 0));
        while let Some((node, depth)) = stack.pop_front() {
            if depth > max_depth {
                continue;
            }
            for succ in self.graph.neighbors(node) {
                if let Some(threshold) = threshold {
                    if euclidean_distance(&self.points[node.index()], &self.points[succ.index()])
                        > threshold
                    {
                        continue;
                    }
                }
                if discovered.visit(succ) {
                    stack.push_back((succ, depth + 1));
                }
            }
        }
        discovered
    }
}

#[pymethods]
impl Graph {
    #[new]
    fn py_new(coords: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Graph::from(&coords))
    }

    fn __repr__(&self) -> String {
        format!(
            "{:?}",
            Dot::with_config(&self.graph, &[Config::EdgeNoLabel, Config::NodeNoLabel])
        )
    }
}

#[pymodule]
fn cev_metrics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;

    #[pyfn(m, name = "confusion")]
    fn confusion_py<'py>(
        py: Python<'py>,
        graph: &Graph,
        codes: PyReadonlyArray1<'_, i16>,
    ) -> &'py PyArray2<u64> {
        let labels = Labels::from_codes(&codes.as_slice().unwrap());
        let confusion = labels.confusion(&graph);
        confusion.counts().into_pyarray(py)
    }

    #[pyfn(m, name="neighborhood", signature=(graph, codes, max_depth=1))]
    fn neighborhood_py<'py>(
        py: Python<'py>,
        graph: &Graph,
        codes: PyReadonlyArray1<'_, i16>,
        max_depth: usize,
    ) -> &'py PyArray2<f64> {
        let labels = Labels::from_codes(&codes.as_slice().unwrap());
        let confusion = labels.confusion(&graph);
        let neighborhood = labels.neighborhood(graph, &confusion, max_depth);
        neighborhood.scores().into_pyarray(py)
    }

    #[pyfn(m, name="confusion_and_neighborhood", signature=(graph, codes, neighborhood_max_depth=1))]
    fn confusion_and_neighborhood_py<'py>(
        py: Python<'py>,
        graph: &Graph,
        codes: PyReadonlyArray1<'_, i16>,
        neighborhood_max_depth: usize,
    ) -> (&'py PyArray2<u64>, &'py PyArray2<f64>) {
        let labels = Labels::from_codes(&codes.as_slice().unwrap());
        let confusion = labels.confusion(&graph);
        let neighborhood = labels.neighborhood(graph, &confusion, neighborhood_max_depth);
        (
            confusion.counts().into_pyarray(py),
            neighborhood.scores().into_pyarray(py),
        )
    }

    Ok(())
}
