use pyo3::prelude::*;
use pyo3::types::PyType;

use delaunator::{triangulate, Point};
use numpy::ndarray::Axis;
use numpy::PyReadonlyArray2;
use petgraph::data::{Element, FromElements};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{NodeIndex, UnGraph};
// use petgraph::visit::Bfs;

fn euclidean_distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

/// A set of labels for a graph
/// Indices of the labels correspond to the indices of the nodes in the graph
/// The labels are represented as integers between 0 and n_categories - 1.
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

    fn n_categories(&self) -> usize {
        self.n_categories
    }

    fn codes(&self) -> &'a [u16] {
        &self.codes
    }
}

#[pyclass(unsendable)]
#[derive(Debug)]
struct Graph {
    graph: UnGraph<usize, f64>,
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
        Self { graph }
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
    fn average_distance_for_labels(&self, labels: &Labels) -> Vec<f64> {
        if self.graph.node_count() != labels.codes().len() {
            panic!("Number of nodes in graph does not match number of labels");
        }
        let mut data = vec![(0.0, 0); labels.n_categories()];
        for edge in self.graph.raw_edges() {
            if labels.codes()[edge.source().index()] == labels.codes()[edge.target().index()] {
                let code = labels.codes()[edge.source().index()] as usize;
                data[code].0 += edge.weight;
                data[code].1 += 1;
            }
        }
        data.iter()
            .map(|(total, count)| *total / *count as f64)
            .collect()
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
