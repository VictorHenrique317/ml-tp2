use std::vec::Vec;
use crate::weighted_data::WeightedData;
use pyo3::prelude::*;

/// Representa um classificador fraco, no caso um decision stump. Escolhi os decision stumps
/// pois foram eles que foram utilizados na aula de boosting, além disso eles são bastante simples.
#[pyclass]
#[derive(Clone)]
pub struct WeakLearner {
    /// Indice da feature que será utilizada como critério de decisão.
    pub feature_index: usize,
    /// Valor da feature que será utilizado como critério de decisão.
    pub feature_target_value: i64,
    /// Predição do classificador fraco, pode ser 1 ou -1.
    pub prediction: i64,

    /// Taxa de aprendizado do classificador fraco.
    pub learning_rate: f64,
    /// Erro do classificador fraco.
    pub error: f64,
    /// Importância (alpha) do classificador fraco.
    pub alpha: f64,
}

#[pymethods]
impl WeakLearner {
    /// Cria um novo classificador fraco vazio, incapaz de classificar qualquer dado.
    /// 
    /// # Returns
    /// Novo classificador fraco.
    #[new]
    pub fn new(learning_rate: f64) -> WeakLearner {
        WeakLearner {
            feature_index: usize::MAX,
            feature_target_value: 0,
            prediction: 0,
            learning_rate: learning_rate,
            error: 0.0,
            alpha: 0.0,
        }
    }

    /// Treina o classificador fraco com um conjunto de samples.
    /// O classificador fraco é treinado escolhendo a feature e o valor da feature que minimizam
    /// o erro ao se fazer a predição (positiva ou negativa) levando somente isso em consideração. 
    /// 
    /// O erro de classificação  é definido como a soma dos pesos das samples que foram classificadas 
    /// incorretamente. Essa busca é exaustiva e testa todas as features com todos os valores possíveis.
    /// 
    /// Ao final o erro do classificador e o alpha são calculados. O alpha é definido como 
    /// learning_rate * log_10((1 - error) / error), onde learning_rate é um hiperparâmetro que regula a 
    /// velocidade com que o boosting convergirá. Quanto maior o learning_rate maior vai ser a diferença
    /// entre os pesos das samples classificadas corretamente e incorretamente na hora de atualizar os pesos.
    /// 
    /// # Arguments
    /// * `weighted_data` - Conjunto de samples.
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
        self.alpha = self.learning_rate * ((1.0 - self.error)/ self.error).log10();
    }

    /// Faz a classificação de um dado com base no valor da feature definido
    /// no treinamento. A predição é feita com base na votação de diferentes classificadores fracos,
    /// sendo o valor final -1 se a soma das diferentes predições ponderadas por alpha for negativa
    /// e 1 caso contrário.
    /// 
    /// # Arguments
    /// * `features` - Features do dado a ser classificado.
    /// 
    /// # Panics
    /// Se o classificador fraco não foi treinado.
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
