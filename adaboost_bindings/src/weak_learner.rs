use std::vec::Vec;
use crate::weighted_data::WeightedData;
use crate::sample::Sample;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct WeakLearner {
    pub feature_index: usize,
    pub threshold: f64,
    pub polarity: i32,
}

#[pymethods]
impl WeakLearner {
    #[new]
    pub fn new() -> WeakLearner {
        WeakLearner {
            feature_index: 0,
            threshold: 0.0,
            polarity: 1,
        }
    }

    pub fn fit(&mut self, weighted_data: WeightedData) {
        let n_features = weighted_data.samples[0].features.len();
        let n_classes = weighted_data.samples.iter().map(|s| s.label).max().unwrap() + 1;
        let mut min_error = f64::INFINITY;

        for feature_index in 0..n_features {
            for threshold in [0.0, 1.0].iter() {
                for polarity in [-1, 1].iter() {
                    let mut error = 0.0;
                    let mut class_counts = vec![0; n_classes as usize];
                    for sample in &weighted_data.samples {
                        let predicted_label = if (sample.features[feature_index] - threshold) * (*polarity as f64) >= 0.0 { 1 } else { 0 };
                        if predicted_label != sample.label {
                            error += sample.getWeight();
                        }
                        class_counts[predicted_label as usize] += 1;
                    }

                    if error < min_error {
                        min_error = error;
                        self.feature_index = feature_index;
                        self.threshold = *threshold;
                        self.polarity = *polarity;
                    }
                }
            }
        }
    }

    pub fn predict(&self, features: Vec<f64>) -> i32 {
        if (features[self.feature_index] - self.threshold) * (self.polarity as f64) >= 0.0 { 1 } else { 0 }
    }
}
