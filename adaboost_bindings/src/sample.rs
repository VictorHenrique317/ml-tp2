use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Sample {
    pub features: Vec<i64>,
    pub label: i64,
    weight: f64,
}
#[pymethods]
impl Sample {
    #[new]
    pub fn new(features: Vec<i64>, label: i64, weight: f64) -> Sample {
        Sample { features, label, weight }
    }

    pub fn getWeight(&self) -> f64 {
        self.weight
    }

    pub fn setWeight(&mut self, weight: f64) {
        self.weight = weight;
    }
}
