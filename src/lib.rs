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

    fn confusion(&self, graph: &Graph, label: u16, threshold: Option<f64>) -> ConfusionResult {
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
            boundary_edges,
            labels: &self,
        }
    }

    fn confusion_all(&self, graph: &Graph) -> Vec<ConfusionResult> {
        let average_distances = self.average_distances(graph);
        (0..self.n_categories)
            .map(|label| {
                let threshold = average_distances[label];
                self.confusion(graph, label as u16, Some(threshold))
            })
            .collect()
    }

    fn neighborhood(&self, graph: &Graph, label: u16, confusion_results: &[ConfusionResult]) {
        let mut boundary_edges: HashSet<EdgeIndex> = HashSet::new();
        for result in confusion_results {
            boundary_edges.extend(&result.boundary_edges);
        }
        let boundary_edge_distances = boundary_edges.iter().map(|edge_index| {
            todo!();
            // graph.graph.edges(a)
            // graph.raw_edges[edge_index.index()]
        });
        todo!();
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

#[derive(Debug)]
struct ConfusionResult<'a> {
    set: HashSet<NodeIndex>,
    boundary_edges: HashSet<EdgeIndex>,
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

    fn confusion_all(&self) -> () {
        let codes = vec![0, 1, 2, 2];
        let labels = Labels::from_codes(&codes);
        let result: Vec<_> = labels
            .confusion_all(&self)
            .iter()
            .map(|r| r.counts())
            .collect();
        println!("{:?}", result);
    }
}

#[pymodule]
fn cev_graph(_py: Python, m: &PyModule) -> PyResult<()> {
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
