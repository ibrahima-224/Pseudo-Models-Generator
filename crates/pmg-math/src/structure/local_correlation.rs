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

//! Corrélation locale par blocs sans matrice globale.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md` §5.4.
//! Ce module implémente des corrélations locales par blocs, permettant de
//! contrôler les propriétés statistiques sans matrice de covariance globale.
//! L'approche par blocs est adaptée au streaming.
//!
//! ## Propriétés
//!
//! - Corrélation intra-bloc contrôlée ;
//! - Indépendance inter-blocs ;
//! - Mémoire O(bloc × dim) au lieu de O(dim²).

use crate::covariance::{Cholesky, Covariance};
use crate::distribution::Distribution;
use crate::distributions::Normal;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Configuration pour la corrélation locale par blocs.
#[derive(Debug, Clone)]
pub struct LocalCorrelationConfig {
    /// Tailles des blocs.
    block_sizes: Vec<usize>,
    /// Corrélations intra-bloc pour chaque bloc.
    block_correlations: Vec<f64>,
}

impl LocalCorrelationConfig {
    /// Crée une nouvelle configuration de corrélation locale.
    ///
    /// # Entrées
    /// - `block_sizes` : tailles des blocs ;
    /// - `block_correlations` : corrélation intra-bloc pour chaque bloc.
    ///
    /// # Erreurs
    /// - [`MathError::InvalidParameter`] si les vecteurs sont de longueurs différentes ;
    /// - [`MathError::NotPsd`] si une corrélation est invalide pour la taille du bloc.
    pub fn new(block_sizes: Vec<usize>, block_correlations: Vec<f64>) -> MathResult<Self> {
        if block_sizes.len() != block_correlations.len() {
            return Err(MathError::InvalidParameter(
                "block_sizes et block_correlations doivent être de même longueur".into(),
            ));
        }
        // Vérifie que chaque corrélation est valide pour la taille du bloc
        for (i, (&size, &rho)) in block_sizes
            .iter()
            .zip(block_correlations.iter())
            .enumerate()
        {
            if size == 0 {
                return Err(MathError::InvalidParameter(format!(
                    "bloc {i} de taille nulle"
                )));
            }
            if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
                return Err(MathError::InvalidParameter(format!(
                    "corrélation du bloc {i} hors [-1, 1] : {rho}"
                )));
            }
            // Vérifie la condition PSD pour une matrice équicorrélée
            let min_rho = -1.0 / (size as f64 - 1.0);
            if rho < min_rho {
                return Err(MathError::NotPsd(format!(
                    "bloc {i} : ρ = {rho} < {min_rho} (borne PSD équicorrélée)"
                )));
            }
        }
        Ok(Self {
            block_sizes,
            block_correlations,
        })
    }

    /// Retourne les tailles des blocs.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Retourne les corrélations intra-bloc.
    pub fn block_correlations(&self) -> &[f64] {
        &self.block_correlations
    }

    /// Retourne la taille totale (somme des blocs).
    pub fn total_size(&self) -> usize {
        self.block_sizes.iter().sum()
    }
}

/// Structure de corrélation locale par blocs.
///
/// Permet de générer des échantillons avec corrélation intra-bloc
/// et indépendance inter-blocs.
#[derive(Debug, Clone)]
pub struct LocalCorrelation {
    /// Configuration.
    config: LocalCorrelationConfig,
    /// Factorisations de Cholesky pour chaque bloc.
    cholesky_blocks: Vec<Cholesky>,
}

impl LocalCorrelation {
    /// Crée une nouvelle structure de corrélation locale.
    ///
    /// # Entrées
    /// - `config` : configuration de corrélation locale.
    ///
    /// # Erreurs
    /// [`MathError::NotPsd`] si une matrice de bloc n'est pas PSD.
    pub fn new(config: LocalCorrelationConfig) -> MathResult<Self> {
        let mut cholesky_blocks = Vec::with_capacity(config.block_sizes.len());
        for (&size, &rho) in config
            .block_sizes
            .iter()
            .zip(config.block_correlations.iter())
        {
            // Construit la matrice équicorrélée pour le bloc
            let mut sigma = vec![0.0; size * size];
            for i in 0..size {
                for j in 0..size {
                    sigma[i * size + j] = if i == j { 1.0 } else { rho };
                }
            }
            let cov = Covariance::new(sigma, size)?;
            cholesky_blocks.push(cov.cholesky().clone());
        }
        Ok(Self {
            config,
            cholesky_blocks,
        })
    }

    /// Génère des échantillons avec corrélation locale.
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `n` : nombre d'échantillons.
    ///
    /// # Sorties
    /// Vecteur de taille n × total_size contenant les échantillons.
    pub fn generate(&self, rng: &mut DeterministicRng, n: usize) -> MathResult<Vec<f64>> {
        if n == 0 {
            return Err(MathError::InvalidParameter(
                "nombre d'échantillons nul".into(),
            ));
        }
        let total_size = self.config.total_size();
        let mut normal = Normal::new(0.0, 1.0)?;
        let mut samples = Vec::with_capacity(n * total_size);
        let mut z = vec![0.0; total_size];

        for _ in 0..n {
            // Génère z ~ N(0,I)
            for zj in z.iter_mut() {
                *zj = normal.sample(rng);
            }
            // Applique la corrélation par blocs
            let mut offset = 0;
            for (block_idx, cholesky) in self.cholesky_blocks.iter().enumerate() {
                let block_size = self.config.block_sizes()[block_idx];
                // x_block = L_block * z_block
                for i in 0..block_size {
                    let mut acc = 0.0;
                    for j in 0..=i {
                        acc += cholesky.l[i * block_size + j] * z[offset + j];
                    }
                    samples.push(acc);
                }
                offset += block_size;
            }
        }
        Ok(samples)
    }

    /// Retourne la configuration.
    pub fn config(&self) -> &LocalCorrelationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::DeterministicRng;

    #[test]
    fn local_correlation_config_new_valid() {
        let config = LocalCorrelationConfig::new(vec![2, 3], vec![0.5, 0.3]);
        assert!(config.is_ok());
    }

    #[test]
    fn local_correlation_config_new_invalid_length() {
        let config = LocalCorrelationConfig::new(vec![2, 3], vec![0.5]);
        assert!(config.is_err());
    }

    #[test]
    fn local_correlation_config_new_invalid_rho() {
        let config = LocalCorrelationConfig::new(vec![2], vec![1.5]);
        assert!(config.is_err());
    }

    #[test]
    fn local_correlation_new_valid() {
        let config = LocalCorrelationConfig::new(vec![2, 3], vec![0.5, 0.3]).unwrap();
        let corr = LocalCorrelation::new(config);
        assert!(corr.is_ok());
    }

    #[test]
    fn local_correlation_generate() {
        let config = LocalCorrelationConfig::new(vec![2, 2], vec![0.5, 0.5]).unwrap();
        let corr = LocalCorrelation::new(config).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let samples = corr.generate(&mut rng, 100).unwrap();
        assert_eq!(samples.len(), 100 * 4);
        // Vérifie que les échantillons sont finis
        for &x in &samples {
            assert!(x.is_finite());
        }
    }
}
