use std::vec::Vec;
use numpy::ndarray::{Array2, Axis, Array1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArray1};
use crate::sample::Sample;
use crate::weak_learner::WeakLearner;
use crate::weighted_data::WeightedData;
use pyo3::prelude::*;

/// Representa o algoritmo AdaBoost que usa como classificadores fracos decision stumps.
#[pyclass]
pub struct AdaBoost {
    /// Número de classificadores fracos usados para fazer a predição.
    pub n_estimators: usize,
    /// Classificadores fracos usados para fazer a predição.
    pub weak_learners: Vec<WeakLearner>,
}

#[pymethods]
impl AdaBoost {
    /// Cria um novo algoritmo AdaBoost.
    /// 
    /// # Arguments
    /// * `n_estimators` - Número de classificadores fracos usados para fazer a predição.
    /// 
    /// # Returns
    /// Novo algoritmo AdaBoost.
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

    /// Treina o algoritmo AdaBoost clássico.
    /// 
    /// # Arguments
    /// * `x` - Array numpy com as features dos dados de treino.
    /// * `y` - Array numpy com os labels dos dados de treino.
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
            let predicted_label = weak_learner.predict(features.clone());
            
            sign_h += weak_learner.alpha  * predicted_label as f64;
        }

        return sign_h.signum() as i64;
    }

    /// Faz a predição com base em um array numpy. Como no algoritmo clássico a predição é
    /// feita com base na votação de diferentes classificadores fracos, sendo o valor final
    /// 0 se a soma das diferentes predições ponderadas por alpha for negativa e 1 caso contrário.
    /// 
    /// # Arguments
    /// * `x` - Array numpy com as features.
    /// 
    /// # Returns
    /// Array numpy com as predições.
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

        let predictions = PyArray1::from_vec(py, predictions);
        Ok(predictions.into_py(py))
    }
}
