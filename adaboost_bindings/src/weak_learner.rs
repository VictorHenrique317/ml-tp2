use std::vec::Vec;
use crate::weighted_data::WeightedData;
use crate::sample::Sample;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct WeakLearner {
    pub feature_index: usize,
    pub threshold: i64,
    pub polarity: i64,

    pub error: f64,
    pub alpha: f64,
}

#[pymethods]
impl WeakLearner {
    #[new]
    pub fn new() -> WeakLearner {
        WeakLearner {
            feature_index: 0,
            threshold: 0,
            polarity: 1,
            error: 0.5,
            alpha: 0.0,
        }
    }

    pub fn fit(&mut self, weighted_data: WeightedData) {
        let n_features = weighted_data.samples[0].features.len();
        let n_classes = weighted_data.samples.iter().map(|s| s.label).max().unwrap() + 1;
        let mut min_error = f64::INFINITY;

        for feature_index in 0..n_features {
            let min_categorial_feature_value = weighted_data.samples.iter()
                .map(|s| s.features[feature_index])
                .min().unwrap();
            
            let max_categorical_feature_value = weighted_data.samples.iter()
                .map(|s| s.features[feature_index])
                .max().unwrap();

            for threshold in min_categorial_feature_value..=max_categorical_feature_value {
                for polarity in [-1, 1].iter() {

                    let mut error = 0.0;
                    let mut class_counts = vec![0; n_classes as usize];
                    for sample in &weighted_data.samples {
                        let predicted_label = if (sample.features[feature_index] - threshold) * (*polarity) >= 0 { 1 } else { 0 };
                        if predicted_label != sample.label {
                            error += sample.getWeight();
                        }
                        class_counts[predicted_label as usize] += 1;
                    }

                    if error <= min_error {
                        min_error = error;
                        self.feature_index = feature_index;
                        self.threshold = threshold;
                        self.polarity = *polarity;
                    }
                }
            }
        }

        self.error = weighted_data.computeWeightedErrorRate(&self);
        self.alpha = 0.5 * ((1.0 - self.error)/ self.error).log10();
    }

    pub fn predict(&self, features: Vec<i64>) -> i64 {
        if (features[self.feature_index] - self.threshold) * (self.polarity) >= 0 { 1 } else { 0 }
    }
}
