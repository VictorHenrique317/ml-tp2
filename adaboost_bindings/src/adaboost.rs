use std::vec::Vec;
use numpy::ndarray::{Array2, Axis, Array1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArray1};
use crate::sample::Sample;
use crate::weak_learner::WeakLearner;
use crate::weighted_data::WeightedData;
use pyo3::prelude::*;

#[pyclass]
pub struct AdaBoost {
    pub n_estimators: usize,
    pub weak_learners: Vec<WeakLearner>,
}

#[pymethods]
impl AdaBoost {
    #[new]
    pub fn new(n_estimators: usize) -> AdaBoost {
        AdaBoost {
            n_estimators,
            weak_learners: Vec::new(),
        }
    }

    fn extractSamples(&self, x: PyReadonlyArray2<i64>, y: PyReadonlyArray1<i64>) -> Vec<Sample> {
        let x: Array2<i64> = x.as_array().to_owned();
        let y: Vec<i64> = y.as_array().to_owned().to_vec();
        let weight = 1.0 / y.len() as f64;
        let mut samples: Vec<Sample> = Vec::new();

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let features: Vec<i64> = row.iter()
            .map(|x_n| *x_n)
            .collect();
            let label: i64 = y[i];

            let sample = Sample::new(features, label, weight);
            samples.push(sample);
        }

        return samples;
    }

    pub fn fit(&mut self, py: Python, x: PyReadonlyArray2<i64>, y: PyReadonlyArray1<i64>) -> PyResult<()>{
        let samples = self.extractSamples(x, y);
        let mut weighted_data = WeightedData::new(samples);

        for i in 0..self.n_estimators {
            
            let mut weak_learner = WeakLearner::new();
            weak_learner.fit(weighted_data.clone());

            weighted_data.updateWeights(&weak_learner);
            self.weak_learners.push(weak_learner);
        }

        Ok(())
    }

    fn singlePredict(&self, features: Vec<i64>) -> i64 {
        let mut sign_h: f64 = 0.0;
        for weak_learner in self.weak_learners.iter() {
            let mut predicted_label = weak_learner.predict(features.clone());
            if predicted_label == 0 { predicted_label = -1; }
            
            sign_h += weak_learner.alpha  * predicted_label as f64;
        }

        if sign_h > 0.0 {
            return 1;
        } else {
            return 0;
        }
    }

    pub fn predict(&self, py: Python, x: PyReadonlyArray2<i64>) -> PyResult<Py<PyArray1<i64>>> {
        let x: Array2<i64> = x.as_array().to_owned();
        let mut predictions: Vec<i64> = Vec::new();
        
        for row in x.axis_iter(Axis(0)) {
            let features: Vec<i64> = row.iter()
                .map(|x_n| *x_n)
                .collect();
            let predicted_label = self.singlePredict(features);
            predictions.push(predicted_label);
        }

        // let gil = pyo3::Python::acquire_gil();
        // let py = gil.python();
        let predictions = PyArray1::from_vec(py, predictions);
        Ok(predictions.into_py(py))
    }
}
