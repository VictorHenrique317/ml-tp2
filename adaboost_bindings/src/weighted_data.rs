use std::vec::Vec;
use crate::weak_learner::WeakLearner;
use crate::sample::Sample;
use pyo3::prelude::*;

/// Guarda o conjunto de samples, que são os dados com seus pesos e labels.
#[pyclass]
#[derive(Clone)]
pub struct WeightedData {
    /// Conjunto de samples.
    pub samples: Vec<Sample>,
}

#[pymethods]
impl WeightedData {
    /// Cria um novo conjunto de samples.
    /// 
    /// # Arguments
    /// * `samples` - Conjunto de samples.
    /// 
    /// # Returns
    /// Novo conjunto de samples.
    #[new]
    pub fn new(samples: Vec<Sample>) -> WeightedData {
        WeightedData { samples }
    }

    /// Calcula o erro que um classificador fraco comete ao classificar as samples.
    /// O erro é definido como a soma dos pesos das samples que foram classificadas
    /// incorretamente.
    /// 
    /// # Arguments
    /// * `weak_learner` - Classificador fraco.
    /// 
    /// # Returns
    /// Erro do classificador fraco.
    pub fn computeWeightedErrorRate(&self,  weak_learner: &WeakLearner) -> f64 {
        let mut error = 0.0;
        let mut i = 0;
        for sample in &self.samples {
            i+=1;
            let predicted_label = weak_learner.predict(sample.features.clone());
            if predicted_label != sample.label {
                error += sample.getWeight();
            }
        }
        error
    }

    /// Atualiza os pesos das samples de acordo com a predição do classificador fraco.
    /// Se a predição for correta, o peso da sample é multiplicado por e^(-alpha).
    /// Se a predição for incorreta, o peso da sample é multiplicado por e^(alpha).
    /// Os pesos são de todas as samples são normalizados para que a soma deles seja 1.
    pub fn updateWeights(&mut self, weak_learner: &WeakLearner) {
        for sample in &mut self.samples {
            let predicted_label = weak_learner.predict(sample.features.clone());
            if predicted_label == sample.label {
                sample.setWeight(sample.getWeight() * (-weak_learner.alpha).exp());
            } else {
                sample.setWeight(sample.getWeight() * weak_learner.alpha.exp());
            }
        }

        let sum_weights: f64 = self.samples.iter().map(|s| s.getWeight()).sum();
        for sample in &mut self.samples {
            sample.setWeight(sample.getWeight() / sum_weights);
        }
    }
}
