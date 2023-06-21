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
    /// Valor do threshold que para qual a feature será comparada. Se a feature for maior que o
    /// threshold, o dado será classificado como 1, caso contrário, será classificado como 0.
    pub threshold: i64,
    /// Polaridade da decisão. Se for 1, o dado será classificado como 1 se for maior que o threshold,
    /// caso contrário, será classificado como 0. Se for -1, o dado será classificado como 1 se for
    /// menor que o threshold, caso contrário, será classificado como 0.
    pub polarity: i64,

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
            feature_index: -1,
            threshold: 0,
            polarity: 0,
            error: 0.0,
            alpha: 0.0,
        }
    }

    /// Treina o classificador fraco com os dados passados. Nessa função os valores de feature_index,
    /// threshold, polarity são definidos. Esses são escolhidos como sendo a combinação
    /// de valores que minimiza o erro dos dados passados, essa busca é exaustiva e testa todas as possibilidades.
    /// 
    /// O erro é definido como a soma dos pesos das samples que foram classificadas incorretamente.
    /// A classificação é feita utilizando o feature_index, threshold e polarity definidos no treinamento.
    /// Ela é definida como 1 se a feature for maior que o threshold e a polaridade for 1 ou se a feature
    /// for menor que o threshold e a polaridade for -1. Caso contrário, a classificação é definida como 0.
    /// 
    /// Depois de ter os valores de feature_index, threshold, polarity definidos, o erro e o peso do classificador
    /// fraco são calculados.
    /// 
    /// # Arguments
    /// * `weighted_data` - Samples que serão utilizadas para treinar o classificador fraco.
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

    /// Classifica um dado levando em consideração que o classificador fraco já foi treinado.
    /// A classificação é feita utilizando o feature_index, threshold e polarity definidos no treinamento.
    /// Ela é definida como 1 se a feature for maior que o threshold e a polaridade for 1 ou se a feature
    /// for menor que o threshold e a polaridade for -1. Caso contrário, a classificação é definida como 0.
    /// 
    /// # Arguments
    /// * `features` - Vetor de features X do dado que será classificado.
    /// 
    /// # Returns
    /// Classificação do dado, 0 ou 1.
    /// 
    /// # Panics
    /// Se o classificador fraco não foi treinado.
    pub fn predict(&self, features: Vec<i64>) -> i64 {
        if (self.feature_index == -1){
            panic!("WeakLearner not trained");
        }

        if (features[self.feature_index] - self.threshold) * (self.polarity) >= 0 { 1 } else { 0 }
    }
}
