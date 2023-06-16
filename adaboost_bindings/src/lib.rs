pub mod adaboost;
pub mod weak_learner;
pub mod weighted_data;
pub mod sample;
use pyo3::prelude::*;
pub use adaboost::AdaBoost;

#[pymodule]
fn adaboost_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AdaBoost>()?;
    Ok(())
}