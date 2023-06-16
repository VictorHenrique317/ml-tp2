use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: i32,
    weight: f64,
}
#[pymethods]
impl Sample {
    #[new]
    pub fn new(features: Vec<f64>, label: i32, weight: f64) -> Sample {
        Sample { features, label, weight }
    }

    pub fn getWeight(&self) -> f64 {
        self.weight
    }

    pub fn setWeight(&mut self, weight: f64) {
        self.weight = weight;
    }
}
