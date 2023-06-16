use std::vec::Vec;
use numpy::ndarray::{Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use crate::sample::Sample;
use crate::weak_learner::WeakLearner;
use crate::weighted_data::WeightedData;
use pyo3::prelude::*;

#[pyclass]
pub struct AdaBoost {
    pub n_estimators: usize,
    pub weak_learners: Vec<WeakLearner>,
    pub alphas: Vec<f64>,
}

#[pymethods]
impl AdaBoost {
    #[new]
    pub fn new(n_estimators: usize) -> AdaBoost {
        AdaBoost {
            n_estimators,
            weak_learners: Vec::new(),
            alphas: Vec::new(),
        }
    }

    fn extractSamples(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> Vec<Sample> {
        let x: Array2<f64> = x.as_array().to_owned();
        let y: Vec<f64> = y.as_array().to_owned().to_vec();
        let weight = 1.0 / y.len() as f64;
        let mut samples: Vec<Sample> = Vec::new();

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let features: Vec<f64> = row.iter()
            .map(|x_n| *x_n)
            .collect();

            let label: i32 = y[i] as i32;
            let sample = Sample::new(features, label, weight);
            samples.push(sample);
        }

        return samples;
    }

    pub fn fit(&mut self, py: Python, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()>{
        let samples = self.extractSamples(x, y);
        let mut weighted_data = WeightedData::new(samples);

        for _ in 0..self.n_estimators {
            let mut weak_learner = WeakLearner::new();
            weak_learner.fit(weighted_data.clone());
            let error = weighted_data.computeWeightedErrorRate(weak_learner.clone());
            let alpha = 0.5 * (1.0 - error) / error;
            weighted_data.updateWeights(&weak_learner, alpha);
            self.weak_learners.push(weak_learner);
            self.alphas.push(alpha);
        }

        Ok(())
    }

    pub fn predict(&self, py: Python, x: PyReadonlyArray2<f64>) -> i32 { // TODO: Antes o argumento era features: Vec<f64>
        let x: Array2<f64> = x.as_array().to_owned();
        
        let mut class_scores = vec![0.0; 2];
        for (weak_learner, alpha) in self.weak_learners.iter().zip(self.alphas.iter()) {
            let predicted_label = weak_learner.predict(features.clone());
            class_scores[predicted_label as usize] += alpha;
        }
        class_scores.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as i32
    }
}
