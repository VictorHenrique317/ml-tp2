use std::vec::Vec;
use crate::weighted_data::WeightedData;
use crate::sample::Sample;
use pyo3::prelude::*;

/// Representa um classificador fraco, no caso um decision stump. Escolhi os decision stumps
/// pois foram eles que foram utilizados na aula de boosting, além disso eles são bastante simples.
#[pyclass]
#[derive(Clone)]
pub struct WeakLearner {
    /// Indice da feature que será utilizada como critério de decisão.
    pub feature_index: usize,
    pub feature_target_value: i64,
    pub prediction: i64,

    /// Erro do classificador fraco.
    pub error: f64,
    /// Peso do classificador fraco.
    pub alpha: f64,
}

#[pymethods]
impl WeakLearner {
    /// Cria um novo classificador fraco vazio, incapaz de classificar qualquer dado.
    /// # Returns
    /// Novo classificador fraco.
    #[new]
    pub fn new() -> WeakLearner {
        WeakLearner {
            feature_index: usize::MAX,
            feature_target_value: 0,
            prediction: 0,
            error: 0.0,
            alpha: 0.0,
        }
    }

    pub fn fit(&mut self, weighted_data: WeightedData) {
        let n_features = weighted_data.samples[0].features.len();
        let mut min_error = f64::INFINITY;

        for feature_index in 0..n_features {
            let min_categorial_feature_value = weighted_data.samples.iter()
                .map(|s| s.features[feature_index])
                .min().unwrap();
            
            let max_categorical_feature_value = weighted_data.samples.iter()
                .map(|s| s.features[feature_index])
                .max().unwrap();

            for feature_value in min_categorial_feature_value..=max_categorical_feature_value {

                let mut error1 = 0.0;
                let mut error2 = 0.0;
                for sample in &weighted_data.samples{
                    
                    let mut prediction = 1;
                    if sample.features[feature_index] != feature_value {
                        prediction = -1;
                    }
                    if prediction != sample.label {
                        error1 += sample.getWeight();
                    }

                    let mut prediction = -1;
                    if sample.features[feature_index] != feature_value {
                        prediction = 1;
                    }
                    if prediction != sample.label {
                        error2 += sample.getWeight();
                    }
                }

                if error1 < min_error {
                    self.feature_index = feature_index;
                    self.feature_target_value = feature_value;
                    self.prediction = 1;
                    min_error = error1;
                }

                if error2 < min_error {
                    self.feature_index = feature_index;
                    self.feature_target_value = feature_value;
                    self.prediction = -1;
                    min_error = error2;
                }
            }
        }

        self.error = weighted_data.computeWeightedErrorRate(&self);
        self.alpha = 0.5 * ((1.0 - self.error)/ self.error).log10();
    }

    
    pub fn predict(&self, features: Vec<i64>) -> i64 {
        if self.feature_index == usize::MAX{
            panic!("WeakLearner not trained");
        }

        let feature_value = features[self.feature_index];

        if feature_value == self.feature_target_value{
            return self.prediction;
        }

        return self.prediction * -1;
    }
}
