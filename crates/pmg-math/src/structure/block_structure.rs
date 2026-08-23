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

//! Structure par blocs W = diag(W₁, W₂, W₃) avec interactions contrôlées.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md` §5.5.
//! Ce module implémente une structure par blocs où chaque bloc peut avoir
//! ses propres propriétés statistiques, avec des interactions contrôlées
//! entre les blocs.
//!
//! ## Propriétés
//!
//! - Blocs diagonaux indépendants ;
//! - Interactions inter-blocs contrôlées ;
//! - Mémoire O(Σ bloc_i²) pour les matrices de blocs.

use crate::covariance::{Cholesky, Covariance};
use crate::distribution::from_config;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;
use pmg_core::distribution_config::DistributionConfig;

/// Configuration pour un bloc individuel.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// Taille du bloc.
    size: usize,
    /// Configuration de distribution pour les éléments du bloc.
    distribution: DistributionConfig,
    /// Matrice de corrélation intra-bloc (optionnelle, sinon identité).
    intra_correlation: Option<Vec<f64>>,
}

impl BlockConfig {
    /// Crée une nouvelle configuration de bloc.
    ///
    /// # Entrées
    /// - `size` : taille du bloc ;
    /// - `distribution` : configuration de distribution.
    pub fn new(size: usize, distribution: DistributionConfig) -> MathResult<Self> {
        if size == 0 {
            return Err(MathError::InvalidParameter("taille de bloc nulle".into()));
        }
        // Valide que la distribution est utilisable
        let _dist = from_config(&distribution)?;
        Ok(Self {
            size,
            distribution,
            intra_correlation: None,
        })
    }

    /// Ajoute une corrélation intra-bloc.
    ///
    /// # Entrées
    /// - `rho` : vecteur de corrélations (size*(size-1)/2 valeurs).
    ///
    /// # Erreurs
    /// [`MathError::NotPsd`] si la matrice de corrélation n'est pas PSD.
    pub fn with_intra_correlation(mut self, rho: &[f64]) -> MathResult<Self> {
        let expected = self.size.saturating_mul(self.size.saturating_sub(1)) / 2;
        if rho.len() != expected {
            return Err(MathError::InvalidParameter(format!(
                "attendu {expected} corrélations pour taille {}, reçu {}",
                self.size,
                rho.len()
            )));
        }
        // Construit la matrice de corrélation
        let mut sigma = vec![0.0; self.size * self.size];
        for i in 0..self.size {
            sigma[i * self.size + i] = 1.0;
        }
        let mut k = 0;
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                let r = rho[k];
                if !r.is_finite() || !(-1.0..=1.0).contains(&r) {
                    return Err(MathError::InvalidParameter(format!(
                        "corrélation hors [-1, 1] : {r}"
                    )));
                }
                sigma[i * self.size + j] = r;
                sigma[j * self.size + i] = r;
                k += 1;
            }
        }
        // Vérifie que la matrice est PSD
        let _cov = Covariance::new(sigma, self.size)?;
        self.intra_correlation = Some(rho.to_vec());
        Ok(self)
    }

    /// Retourne la taille du bloc.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Retourne la configuration de distribution.
    pub fn distribution(&self) -> &DistributionConfig {
        &self.distribution
    }

    /// Retourne les corrélations intra-bloc si définies.
    pub fn intra_correlation(&self) -> Option<&[f64]> {
        self.intra_correlation.as_deref()
    }
}

/// Structure par blocs W = diag(W₁, W₂, W₃).
///
/// Chaque bloc peut avoir ses propres propriétés statistiques.
#[derive(Debug, Clone)]
pub struct BlockStructure {
    /// Configurations des blocs.
    blocks: Vec<BlockConfig>,
    /// Factorisations de Cholesky pour chaque bloc (si corrélation intra-bloc).
    cholesky_blocks: Vec<Option<Cholesky>>,
}

impl BlockStructure {
    /// Crée une nouvelle structure par blocs.
    ///
    /// # Entrées
    /// - `blocks` : configurations des blocs.
    pub fn new(blocks: Vec<BlockConfig>) -> MathResult<Self> {
        let mut cholesky_blocks = Vec::with_capacity(blocks.len());
        for block in &blocks {
            if let Some(rho) = block.intra_correlation() {
                // Construit la matrice de corrélation
                let size = block.size();
                let mut sigma = vec![0.0; size * size];
                for i in 0..size {
                    sigma[i * size + i] = 1.0;
                }
                let mut k = 0;
                for i in 0..size {
                    for j in (i + 1)..size {
                        let r = rho[k];
                        sigma[i * size + j] = r;
                        sigma[j * size + i] = r;
                        k += 1;
                    }
                }
                let cov = Covariance::new(sigma, size)?;
                cholesky_blocks.push(Some(cov.cholesky().clone()));
            } else {
                cholesky_blocks.push(None);
            }
        }
        Ok(Self {
            blocks,
            cholesky_blocks,
        })
    }

    /// Génère un tenseur de la structure par blocs.
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `shape` : forme du tenseur (doit correspondre à la somme des tailles de blocs).
    ///
    /// # Sorties
    /// Vecteur plat contenant tous les éléments du tenseur.
    pub fn generate(&self, rng: &mut DeterministicRng, shape: &[usize]) -> MathResult<Vec<f64>> {
        let total_elements: usize = shape.iter().product();
        let total_block_size: usize = self.blocks.iter().map(|b| b.size()).sum();
        if total_elements != total_block_size {
            return Err(MathError::InvalidParameter(format!(
                "forme du tenseur {total_elements} ≠ taille totale des blocs {total_block_size}"
            )));
        }

        let mut elements = Vec::with_capacity(total_elements);
        let mut _offset = 0;

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let size = block.size();
            let mut dist = from_config(block.distribution())?;

            if let Some(cholesky) = &self.cholesky_blocks[block_idx] {
                // Génère des éléments corrélés dans le bloc
                let mut z = vec![0.0; size];
                for zj in z.iter_mut() {
                    *zj = dist.sample(rng);
                }
                // Applique la corrélation
                for i in 0..size {
                    let mut acc = 0.0;
                    for (j, z_val) in z.iter().enumerate().take(i + 1) {
                        acc += cholesky.l[i * size + j] * z_val;
                    }
                    elements.push(acc);
                }
            } else {
                // Génère des éléments indépendants
                for _ in 0..size {
                    elements.push(dist.sample(rng));
                }
            }
            _offset += size;
        }
        Ok(elements)
    }

    /// Retourne les configurations des blocs.
    pub fn blocks(&self) -> &[BlockConfig] {
        &self.blocks
    }

    /// Retourne la taille totale.
    pub fn total_size(&self) -> usize {
        self.blocks.iter().map(|b| b.size()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::DeterministicRng;

    #[test]
    fn block_config_new_valid() {
        let config = BlockConfig::new(3, DistributionConfig::normal(0.0, 1.0));
        assert!(config.is_ok());
    }

    #[test]
    fn block_config_new_invalid_size() {
        let config = BlockConfig::new(0, DistributionConfig::normal(0.0, 1.0));
        assert!(config.is_err());
    }

    #[test]
    fn block_config_with_intra_correlation() {
        let config = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0))
            .unwrap()
            .with_intra_correlation(&[0.5]);
        assert!(config.is_ok());
    }

    #[test]
    fn block_structure_new_valid() {
        let block1 = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0)).unwrap();
        let block2 = BlockConfig::new(3, DistributionConfig::normal(0.0, 1.0)).unwrap();
        let structure = BlockStructure::new(vec![block1, block2]);
        assert!(structure.is_ok());
    }

    #[test]
    fn block_structure_generate() {
        let block1 = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0)).unwrap();
        let block2 = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0)).unwrap();
        let structure = BlockStructure::new(vec![block1, block2]).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let shape = vec![4];
        let elements = structure.generate(&mut rng, &shape).unwrap();
        assert_eq!(elements.len(), 4);
        for &x in &elements {
            assert!(x.is_finite());
        }
    }
}
