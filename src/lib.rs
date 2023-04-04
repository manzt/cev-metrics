use pyo3::prelude::*;

use delaunator::{triangulate, Point};
use numpy::ndarray::Axis;
use numpy::PyReadonlyArray2;
use petgraph::data::{Element, FromElements};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, VisitMap};
use std::collections::{HashSet, VecDeque};

fn euclidean_distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

/// A set of labels for a graph
/// Indices of the labels correspond to the indices of the nodes in the graph
/// The labels are represented as integers between 0 and n_categories - 1.
#[derive(Debug)]
struct Labels<'a> {
    codes: &'a [u16],
    n_categories: usize,
}

#[derive(Debug)]
struct ConfusionResult<'a> {
    set: HashSet<NodeIndex>,
    boundaries: HashSet<EdgeIndex>,
    labels: &'a Labels<'a>,
}

impl<'a> ConfusionResult<'a> {
    fn counts(&self) -> Vec<u64> {
        let mut v = vec![0; self.labels.n_categories];
        for node in &self.set {
            let code = self.labels.codes[node.index()];
            v[code as usize] += 1;
        }
        v
    }

    fn contains(&self, node: &NodeIndex) -> bool {
        self.set.contains(&node)
    }
}

trait Matrix {
    fn counts(&self) -> Vec<Vec<u64>>;
}

impl<'a> Matrix for Vec<ConfusionResult<'a>> {
    fn counts(&self) -> Vec<Vec<u64>> {
        self.iter().map(|r| r.counts()).collect()
    }
}

#[derive(Debug)]
struct NeighborhoodResult {}

impl<'a> Labels<'a> {
    fn from_codes(codes: &'a [u16]) -> Self {
        let max = *codes.iter().max().unwrap();
        Self {
            codes,
            n_categories: (max + 1) as usize,
        }
    }

    fn average_distances(&self, graph: &Graph) -> Vec<f64> {
        if graph.graph.node_count() != self.codes.len() {
            panic!("Number of nodes in graph does not match number of self");
        }
        let mut data = vec![(0.0, 0); self.n_categories];
        for edge in graph.graph.raw_edges() {
            if self.codes[edge.source().index()] == self.codes[edge.target().index()] {
                let code = self.codes[edge.source().index()] as usize;
                data[code].0 += edge.weight;
                data[code].1 += 1;
            }
        }
        data.iter()
            .map(|(total, count)| *total / *count as f64)
            .collect()
    }

    fn confusion_for_label(
        &self,
        graph: &Graph,
        label: u16,
        threshold: Option<f64>,
    ) -> ConfusionResult {
        let mut visited_with_threshold = HashSet::<NodeIndex>::new();
        let mut visited_without_threshold = HashSet::<NodeIndex>::new();

        for node in graph.graph.node_indices() {
            if self.codes[node.index()] != label {
                continue;
            }
            let inner = graph.bfs(node, 1, threshold);
            let outer = graph.bfs(node, 2, None);
            visited_with_threshold.extend(&inner);
            visited_without_threshold.extend(outer.intersection(&inner));
        }

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
        let average_distances = self.average_distances(graph);
        (0..self.n_categories)
            .map(|label| {
                let threshold = average_distances[label];
                self.confusion_for_label(graph, label as u16, Some(threshold))
            })
            .collect()
    }

    fn neighborhood_for_label(
        &self,
        graph: &Graph,
        counfusion_result: &ConfusionResult,
        max_depth: usize,
    ) -> Vec<(usize, f64)> {
        let boundary_distances = counfusion_result.boundaries.iter().map(|edge_index| {
            let edge = &graph.graph.raw_edges()[edge_index.index()];
            let label = self.codes[edge.target().index()];
            (label as usize, edge.weight)
        });

        let mut visited = HashSet::new();
        for edge_index in &counfusion_result.boundaries {
            let edge = &graph.graph.raw_edges()[edge_index.index()];
            for target in graph.bfs(edge.target(), max_depth, None) {
                if counfusion_result.contains(&target) {
                    continue;
                }
                visited.insert((edge.source(), target));
            }
        }

        let connections = visited.iter().map(|(source, target)| {
            let label = self.codes[target.index()];
            let distance =
                euclidean_distance(&graph.points[source.index()], &graph.points[target.index()]);
            (label as usize, distance)
        });

        boundary_distances.chain(connections).collect()
    }

    fn neighborhood(
        &self,
        graph: &Graph,
        counfusion_results: &Vec<ConfusionResult>,
        max_depth: Option<usize>,
    ) -> Vec<Vec<(usize, f64)>> {
        counfusion_results
            .iter()
            .map(|c| self.neighborhood_for_label(graph, c, max_depth.unwrap_or(1)))
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
            Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
        )
    }

    fn run(&self) -> () {
        let codes = vec![0, 1, 2, 2];
        let labels = Labels::from_codes(&codes);
        let confusion = labels.confusion(&self);
        println!("{:?}", confusion.counts());
        let neighborhood = labels.neighborhood(&self, &confusion, None);
        println!("{:?}", neighborhood);
    }
}

#[pymodule]
fn cev_metrics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 4;
        assert_eq!(result, 4);
    }
}
