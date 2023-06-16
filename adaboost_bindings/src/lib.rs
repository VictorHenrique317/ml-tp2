pub mod adaboost;
pub mod weak_learner;
pub mod weighted_data;
pub mod sample;
use pyo3::prelude::*;
pub use adaboost::AdaBoost;
// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

// /// A Python module implemented in Rust.
#[pymodule]
fn adaboost_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AdaBoost>()?;
    Ok(())
}