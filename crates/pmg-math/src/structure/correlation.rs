// Copyright (C) 2024 PMG Contributors
// This file is part of PMG (Pseudo-Model Generator).
//
// PMG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PMG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PMG.  If not, see <https://www.gnu.org/licenses/>.

//! Corrélation contrôlée globale via matrice de covariance.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md` §5.3.
//! Ce module implémente la corrélation contrôlée selon le modèle x = Az avec
//! z ~ N(0,I) et Σ = AAᵀ. La matrice Σ doit être symétrique et semi-définie
//! positive.
//!
//! ## Propriétés
//!
//! - Corrélation globale entre toutes les dimensions ;
//! - Utilise la factorisation de Cholesky pour la génération ;
//! - Vérification de PSD explicite.

use crate::covariance::{Cholesky, Covariance};
use crate::distribution::Distribution;
use crate::distributions::Normal;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Configuration de corrélation globale.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Matrice de covariance cible Σ (dim × dim, stockée ligne par ligne).
    sigma: Vec<f64>,
    /// Dimension de la matrice.
    dim: usize,
}

impl CorrelationConfig {
    /// Crée une nouvelle configuration de corrélation.
    ///
    /// # Entrées
    /// - `sigma` : matrice de covariance (dim × dim) ;
    /// - `dim` : dimension.
    ///
    /// # Erreurs
    /// - [`MathError::InvalidParameter`] si la matrice n'est pas carrée ;
    /// - [`MathError::NotPsd`] si la matrice n'est pas PSD.
    pub fn new(sigma: Vec<f64>, dim: usize) -> MathResult<Self> {
        // Valide que la matrice est PSD via la construction Covariance
        let _cov = Covariance::new(sigma.clone(), dim)?;
        Ok(Self { sigma, dim })
    }

    /// Crée une configuration de corrélation à partir de corrélations par paires.
    ///
    /// # Entrées
    /// - `dim` : dimension ;
    /// - `rho` : vecteur de corrélations (dim*(dim-1)/2 valeurs).
    ///
    /// # Erreurs
    /// - [`MathError::InvalidParameter`] si le nombre de corrélations est incorrect ;
    /// - [`MathError::NotPsd`] si la matrice résultante n'est pas PSD.
    pub fn from_pairwise(dim: usize, rho: &[f64]) -> MathResult<Self> {
        let cov = Covariance::from_pairwise_correlations(dim, rho)?;
        let sigma = cov.sigma().to_vec();
        Ok(Self { sigma, dim })
    }

    /// Accède à la matrice de covariance.
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Retourne la dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Structure de corrélation globale.
///
/// Permet de générer des échantillons corrélés selon x = Az avec z ~ N(0,I).
#[derive(Debug, Clone)]
pub struct Correlation {
    /// Configuration de corrélation.
    config: CorrelationConfig,
    /// Factorisation de Cholesky de Σ.
    cholesky: Cholesky,
}

impl Correlation {
    /// Crée une nouvelle structure de corrélation.
    ///
    /// # Entrées
    /// - `config` : configuration de corrélation.
    ///
    /// # Erreurs
    /// [`MathError::NotPsd`] si la matrice n'est pas PSD.
    pub fn new(config: CorrelationConfig) -> MathResult<Self> {
        let cov = Covariance::new(config.sigma.clone(), config.dim)?;
        let cholesky = cov.cholesky().clone();
        Ok(Self { config, cholesky })
    }

    /// Génère un échantillon corrélé.
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `n` : nombre d'échantillons.
    ///
    /// # Sorties
    /// Vecteur de taille n × dim contenant les échantillons.
    pub fn generate(&self, rng: &mut DeterministicRng, n: usize) -> MathResult<Vec<f64>> {
        if n == 0 {
            return Err(MathError::InvalidParameter(
                "nombre d'échantillons nul".into(),
            ));
        }
        let dim = self.config.dim;
        let mut normal = Normal::new(0.0, 1.0)?;
        let mut samples = Vec::with_capacity(n * dim);
        let mut z = vec![0.0; dim];

        for _ in 0..n {
            // Génère z ~ N(0,I)
            for zj in z.iter_mut() {
                *zj = normal.sample(rng);
            }
            // Calcule x = L z où L est la factorisation de Cholesky
            for i in 0..dim {
                let mut acc = 0.0;
                for (j, z_val) in z.iter().enumerate().take(i + 1) {
                    acc += self.cholesky.l[i * dim + j] * z_val;
                }
                samples.push(acc);
            }
        }
        Ok(samples)
    }

    /// Vérifie que la matrice de covariance est symétrique et PSD.
    pub fn validate(&self) -> MathResult<()> {
        // La validation est déjà faite lors de la construction
        Ok(())
    }

    /// Retourne la configuration.
    pub fn config(&self) -> &CorrelationConfig {
        &self.config
    }

    /// Retourne la factorisation de Cholesky.
    pub fn cholesky(&self) -> &Cholesky {
        &self.cholesky
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::DeterministicRng;

    #[test]
    fn correlation_config_new_valid() {
        // Matrice identité 2x2
        let sigma = vec![1.0, 0.0, 0.0, 1.0];
        let config = CorrelationConfig::new(sigma, 2);
        assert!(config.is_ok());
    }

    #[test]
    fn correlation_config_new_not_psd() {
        // Matrice non PSD : [[1, 2], [2, 1]]
        let sigma = vec![1.0, 2.0, 2.0, 1.0];
        let config = CorrelationConfig::new(sigma, 2);
        assert!(config.is_err());
    }

    #[test]
    fn correlation_config_from_pairwise() {
        // Matrice de corrélation 2x2 avec ρ=0.5
        let rho = vec![0.5];
        let config = CorrelationConfig::from_pairwise(2, &rho).unwrap();
        assert_eq!(config.dim(), 2);
    }

    #[test]
    fn correlation_new_valid() {
        let sigma = vec![1.0, 0.5, 0.5, 1.0];
        let config = CorrelationConfig::new(sigma, 2).unwrap();
        let corr = Correlation::new(config);
        assert!(corr.is_ok());
    }

    #[test]
    fn correlation_generate() {
        let sigma = vec![1.0, 0.5, 0.5, 1.0];
        let config = CorrelationConfig::new(sigma, 2).unwrap();
        let corr = Correlation::new(config).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let samples = corr.generate(&mut rng, 100).unwrap();
        assert_eq!(samples.len(), 100 * 2);
        // Vérifie la corrélation empirique
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        for i in 0..100 {
            let x = samples[i * 2];
            let y = samples[i * 2 + 1];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        let n = 100.0;
        let corr_emp = (n * sum_xy - sum_x * sum_y)
            / ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        // La corrélation devrait être proche de 0.5
        assert!(
            (corr_emp - 0.5).abs() < 0.2,
            "corrélation empirique {corr_emp} loin de 0.5"
        );
    }
}
