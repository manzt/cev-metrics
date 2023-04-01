use delaunator::{triangulate, Point};
use numpy::ndarray::Axis;
use numpy::PyReadonlyArray2;
use petgraph::{graph::UnGraph, adj::NodeIndex};
use petgraph::visit::Bfs;
use pyo3::{
    pymodule,
    types::PyModule,
    PyObject, PyResult, Python,
};

fn create_graph(points: &Vec<Point>) -> UnGraph<usize, f64> {
    let tris = triangulate(&points).triangles;
    let mut graph = UnGraph::<usize, f64>::new_undirected();
    for (&u, &v) in tris.iter().zip(tris.iter().skip(1)) {
        let a = graph.add_node(u);
        let b = graph.add_node(v);
        let distance =
            ((points[u].x - points[v].x).powi(2) + (points[u].y - points[v].y).powi(2)).sqrt();
        graph.add_edge(a, b, distance);
    }
    graph
}

// fn bfs(graph: &UnGraph<usize, f64>) -> () {
//     let mut bfs = Bfs::new(&graph, NodeIndex::new(0));
//     while let Some(nx) = bfs.next(&graph) {
//         println!("Next: {:?}", nx);
//     };
// }

#[pymodule]
fn cev_graph(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "graph")]
    fn graph_py(_py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<PyObject> {
        let points: Vec<_> = x
            .as_array()
            .lanes(Axis(1))
            .into_iter()
            .map(|x| Point { x: x[0], y: x[1] })
            .collect();
        let graph = create_graph(&points);
        println!("{:?}", graph);
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
