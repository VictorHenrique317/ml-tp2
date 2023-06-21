use pyo3::prelude::*;

/// Representa um dado, com suas features, label, e seu peso.
/// As features são representadas por um vetor de inteiros, e o label só pode ser
/// 1 ou 0.
#[derive(Clone, Debug)]
#[pyclass]
pub struct Sample {
    /// Vetor de features X do dado.
    pub features: Vec<i64>,
    /// Label y do dado, que só pode ser 1 ou 0.
    pub label: i64,
    /// Peso do dado.
    weight: f64,
}
#[pymethods]
impl Sample {
    /// Cria um novo dado.
    /// 
    /// # Arguments
    /// * `features` - Vetor de features X do dado.
    /// * `label` - Label y do dado.
    /// * `weight` - Peso do dado.
    /// 
    /// # Returns
    /// Novo dado.
    #[new]
    pub fn new(features: Vec<i64>, label: i64, weight: f64) -> Sample {
        if label != 0 && label != 1 {
            panic!("Label must be 0 or 1");
        }
        Sample { features, label, weight }
    }

    /// Retorna o peso do dado.
    /// 
    /// # Returns
    /// Peso do dado.
    pub fn getWeight(&self) -> f64 {
        self.weight
    }

    /// Define o peso do dado.
    /// 
    /// # Arguments
    /// * `weight` - Peso do dado.
    pub fn setWeight(&mut self, weight: f64) {
        self.weight = weight;
    }
}
