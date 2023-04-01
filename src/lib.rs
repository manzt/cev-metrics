use delaunator::{triangulate, Point};
use numpy::ndarray::Axis;
use numpy::PyReadonlyArray2;
use petgraph::data::{Element, FromElements};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::Bfs;
use pyo3::{pymodule, types::PyModule, PyObject, PyResult, Python};

#[derive(Debug)]
struct CevGraph {
    graph: UnGraph<usize, f64>,
}

impl From<&Vec<Point>> for CevGraph {
    fn from(points: &Vec<Point>) -> Self {
        println!("{:?}", points);

        let triangulation = triangulate(&points);
        println!("{:?}", triangulation.triangles);

        let nodes = std::iter::repeat(Element::Node { weight: 0 }).take(points.len());
        let mut graph = UnGraph::<_, _>::from_elements(nodes);

        for triangle in triangulation.triangles.chunks(3) {
            let (a, b, c) = (triangle[0], triangle[1], triangle[2]);
            // TODO: Avoid adding duplicate edges
            graph.add_edge(
                NodeIndex::new(a),
                NodeIndex::new(b),
                euclidean_distance(&points[a], &points[b]),
            );
            graph.add_edge(
                NodeIndex::new(b),
                NodeIndex::new(c),
                euclidean_distance(&points[b], &points[c]),
            );
            graph.add_edge(
                NodeIndex::new(c),
                NodeIndex::new(a),
                euclidean_distance(&points[c], &points[a]),
            );
        }

        Self { graph }
    }
}

impl From<&PyReadonlyArray2<'_, f64>> for CevGraph {
    fn from(x: &PyReadonlyArray2<'_, f64>) -> Self {
        let points: Vec<_> = x
            .as_array()
            .lanes(Axis(1))
            .into_iter()
            .map(|x| Point { x: x[0], y: x[1] })
            .collect();
        Self::from(&points)
    }
}

fn euclidean_distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

#[pymodule]
fn cev_graph(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "graph")]
    fn graph_py(_py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<PyObject> {
        let graph = CevGraph::from(&x);
        println!(
            "{:?}",
            Dot::with_config(&graph.graph, &[Config::EdgeNoLabel])
        );
        Ok(_py.None())
    }
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
