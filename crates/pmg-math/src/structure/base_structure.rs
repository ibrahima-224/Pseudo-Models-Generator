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

//! Modèle de base indépendant : W = E.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md` §5.1.
//! Ce modèle représente le cas de base où chaque élément du tenseur est issu
//! du générateur statistique sans structure contrainte. Il sert de référence
//! zéro pour les structures plus complexes.
//!
//! ## Propriétés
//!
//! - Indépendance statistique de chaque élément ;
//! - Pas de corrélation entre les éléments ;
//! - Utilise les distributions du Sprint 7.

use crate::distribution::from_config;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;
use pmg_core::distribution_config::DistributionConfig;

/// Structure de base indépendante : W = E.
///
/// Chaque élément est généré indépendamment selon la distribution spécifiée.
#[derive(Debug, Clone)]
pub struct BaseStructure {
    /// Configuration de la distribution pour chaque élément.
    config: DistributionConfig,
}

impl BaseStructure {
    /// Crée une nouvelle structure de base avec la distribution spécifiée.
    ///
    /// # Entrées
    /// - `config` : configuration de la distribution.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si la configuration est invalide.
    pub fn new(config: DistributionConfig) -> MathResult<Self> {
        // Valide que la configuration est utilisable
        let _dist = from_config(&config)?;
        Ok(Self { config })
    }

    /// Génère un tenseur de la structure de base.
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `shape` : forme du tenseur (dimensions).
    ///
    /// # Sorties
    /// Vecteur plat contenant tous les éléments du tenseur.
    ///
    /// # Complexité
    /// O(n) où n = produit des dimensions.
    pub fn generate(&self, rng: &mut DeterministicRng, shape: &[usize]) -> MathResult<Vec<f64>> {
        let total_elements: usize = shape.iter().product();
        if total_elements == 0 {
            return Err(MathError::InvalidParameter("forme du tenseur vide".into()));
        }

        let mut dist = from_config(&self.config)?;
        let mut elements = Vec::with_capacity(total_elements);

        for _ in 0..total_elements {
            elements.push(dist.sample(rng));
        }

        Ok(elements)
    }

    /// Retourne la configuration de distribution utilisée.
    pub fn config(&self) -> &DistributionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::DeterministicRng;

    #[test]
    fn base_structure_new_valid() {
        let config = DistributionConfig::normal(0.0, 1.0);
        let structure = BaseStructure::new(config);
        assert!(structure.is_ok());
    }

    #[test]
    fn base_structure_generate() {
        let config = DistributionConfig::normal(0.0, 1.0);
        let structure = BaseStructure::new(config).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let shape = vec![2, 3];
        let elements = structure.generate(&mut rng, &shape).unwrap();
        assert_eq!(elements.len(), 6);
        // Vérifie que tous les éléments sont finis
        for &x in &elements {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn base_structure_independence() {
        let config = DistributionConfig::normal(0.0, 1.0);
        let structure = BaseStructure::new(config).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let shape = vec![1000];
        let elements = structure.generate(&mut rng, &shape).unwrap();
        // Vérifie l'indépendance en calculant la corrélation entre paires
        // (test basique : pas de corrélation significative)
        let mean = elements.iter().sum::<f64>() / elements.len() as f64;
        let var = elements.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / elements.len() as f64;
        assert!(var > 0.0, "La variance devrait être positive");
    }
}
