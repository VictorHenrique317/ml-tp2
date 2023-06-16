use std::vec::Vec;
use crate::weak_learner::WeakLearner;
use crate::sample::Sample;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct WeightedData {
    pub samples: Vec<Sample>,
}

#[pymethods]
impl WeightedData {
    #[new]
    pub fn new(samples: Vec<Sample>) -> WeightedData {
        WeightedData { samples }
    }

    pub fn computeWeightedErrorRate(&self, weak_learner: WeakLearner) -> f64 {
        let mut error = 0.0;
        for sample in &self.samples {
            let predicted_label = weak_learner.predict(sample.features.clone());
            if predicted_label != sample.label {
                error += sample.getWeight();
            }
        }
        error
    }

    pub fn updateWeights(&mut self, weak_learner: &WeakLearner, alpha: f64) {
        for sample in &mut self.samples {
            let predicted_label = weak_learner.predict(sample.features.clone());
            if predicted_label == sample.label {
                sample.setWeight(sample.getWeight() * (-alpha).exp());
            } else {
                sample.setWeight(sample.getWeight() * alpha.exp());
            }
        }

        let sum_weights: f64 = self.samples.iter().map(|s| s.getWeight()).sum();
        for sample in &mut self.samples {
            sample.setWeight(sample.getWeight() / sum_weights);
        }
    }
}
