use std::vec::Vec;
use crate::weak_learner::WeakLearner;
use crate::weighted_data::WeightedData;
use pyo3::prelude::*;

#[pyclass]
pub struct AdaBoost {
    pub weak_learners: Vec<WeakLearner>,
    pub alphas: Vec<f64>,
}

#[pymethods]
impl AdaBoost {
    #[new]
    pub fn new() -> AdaBoost {
        AdaBoost {
            weak_learners: Vec::new(),
            alphas: Vec::new(),
        }
    }

    pub fn fit(&mut self, weighted_data: WeightedData, n_estimators: usize) {
        let mut weighted_data = weighted_data;

        for _ in 0..n_estimators {
            let mut weak_learner = WeakLearner::new();
            weak_learner.fit(weighted_data.clone());
            let error = weighted_data.computeWeightedErrorRate(weak_learner.clone());
            let alpha = 0.5 * (1.0 - error) / error;
            weighted_data.updateWeights(&weak_learner, alpha);
            self.weak_learners.push(weak_learner);
            self.alphas.push(alpha);
        }
    }

    pub fn predict(&self, features: Vec<f64>) -> i32 {
        let mut class_scores = vec![0.0; 2];
        for (weak_learner, alpha) in self.weak_learners.iter().zip(self.alphas.iter()) {
            let predicted_label = weak_learner.predict(features.clone());
            class_scores[predicted_label as usize] += alpha;
        }
        class_scores.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as i32
    }
}
