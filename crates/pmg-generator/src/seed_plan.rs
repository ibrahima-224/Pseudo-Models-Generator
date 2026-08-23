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

//! Plan de seed hiérarchique pour la génération déterministe.
//!
//! Ce module fournit les primitives de dérivation de seed pour les tenseurs
//! et les chunks, garantissant que la génération complète ou par chunks
//! produit les mêmes résultats.
//!
//! # Hiérarchie des seeds
//!
//! ```text
//! Seed globale (S_global)
//!    ↓
//! Seed tenseur (S_tensor) = H(S_global, model_id, tensor_name, layer_id, version)
//!    ↓
//! Seed chunk (S_chunk) = H(S_tensor, "chunk", chunk_id)
//! ```
//!
//! Cette hiérarchie garantit :
//! - Indépendance des tenseurs (chaque tenseur a son propre flux) ;
//! - Indépendance des chunks (chaque chunk a son propre flux dérivé) ;
//! - Déterminisme strict (mêmes entrées ⇒ mêmes sorties).

use pmg_math::rng::{derive_seed, derive_sub_seed, DeterministicRng, SeedPlan};

use crate::error::{GeneratorError, GeneratorResult};

/// Plan de seed du générateur pour un tenseur donné.
///
/// Encapsule les paramètres de dérivation et fournit les seeds dérivées
/// pour les chunks. Le plan de seed garantit la reproductibilité de la génération.
///
/// # Exemple
///
/// ```
/// use pmg_generator::GeneratorSeedPlan;
///
/// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
/// assert_eq!(plan.seed_global, 42);
/// assert_eq!(plan.model_id, "glm-5.2");
/// ```
#[derive(Debug, Clone)]
pub struct GeneratorSeedPlan {
    /// Seed globale du processus de génération.
    pub seed_global: u64,
    /// Identifiant du modèle (ex. `"glm-5.2"`).
    pub model_id: String,
    /// Version du générateur.
    pub generation_version: String,
}

impl GeneratorSeedPlan {
    /// Crée un nouveau plan de seed.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GeneratorSeedPlan;
    ///
    /// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    /// assert!(plan.validate().is_ok());
    /// ```
    pub fn new(
        seed_global: u64,
        model_id: impl Into<String>,
        generation_version: impl Into<String>,
    ) -> Self {
        Self {
            seed_global,
            model_id: model_id.into(),
            generation_version: generation_version.into(),
        }
    }

    /// Valide que la seed globale est non nulle.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la seed globale est nulle.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GeneratorSeedPlan;
    ///
    /// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    /// assert!(plan.validate().is_ok());
    ///
    /// let invalid_plan = GeneratorSeedPlan::new(0, "glm-5.2", "1.0.0");
    /// assert!(invalid_plan.validate().is_err());
    /// ```
    pub fn validate(&self) -> GeneratorResult<()> {
        if self.seed_global == 0 {
            return Err(GeneratorError::InvalidModelConfig(
                "seed globale nulle interdite".into(),
            ));
        }
        Ok(())
    }

    /// Dérive la seed d'un tenseur à partir de son nom et de son index de couche.
    ///
    /// # Paramètres
    /// - `tensor_name` : nom complet du tenseur (ex. `"model.layers.0.mlp.gate.weight"`)
    /// - `layer_id` : index de couche (0-based), `None` pour les tenseurs hors couches
    ///
    /// # Retourne
    /// Les 32 octets de la seed dérivée.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GeneratorSeedPlan;
    ///
    /// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    /// let seed = plan.derive_tensor_seed("model.layers.0.mlp.gate.weight", Some(0));
    /// assert_eq!(seed.len(), 32);
    /// ```
    pub fn derive_tensor_seed(&self, tensor_name: &str, layer_id: Option<u32>) -> [u8; 32] {
        let plan = SeedPlan {
            seed_global: self.seed_global,
            model_id: &self.model_id,
            tensor_name,
            layer_id,
            generation_version: &self.generation_version,
        };
        derive_seed(&plan)
    }

    /// Dérive la seed d'un chunk à partir de la seed du tenseur.
    ///
    /// # Paramètres
    /// - `tensor_seed` : seed du tenseur (32 octets)
    /// - `chunk_id` : index du chunk (0-based)
    ///
    /// # Retourne
    /// Les 32 octets de la seed dérivée pour le chunk.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GeneratorSeedPlan;
    ///
    /// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    /// let tensor_seed = plan.derive_tensor_seed("tensor", Some(0));
    /// let chunk_seed = GeneratorSeedPlan::derive_chunk_seed(&tensor_seed, 0);
    /// assert_eq!(chunk_seed.len(), 32);
    /// ```
    pub fn derive_chunk_seed(tensor_seed: &[u8; 32], chunk_id: u64) -> [u8; 32] {
        // Conversion u64 → u32 pour la compatibilité avec derive_sub_seed
        derive_sub_seed(tensor_seed, "chunk", chunk_id as u32)
    }

    /// Crée un RNG déterministe pour un tenseur.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GeneratorSeedPlan;
    ///
    /// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    /// let mut rng = plan.tensor_rng("model.layers.0.mlp.gate.weight", Some(0));
    /// let value = rng.next_f64();
    /// ```
    pub fn tensor_rng(&self, tensor_name: &str, layer_id: Option<u32>) -> DeterministicRng {
        let seed = self.derive_tensor_seed(tensor_name, layer_id);
        DeterministicRng::from_seed(seed)
    }

    /// Crée un RNG déterministe pour un chunk.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GeneratorSeedPlan;
    ///
    /// let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    /// let tensor_seed = plan.derive_tensor_seed("tensor", Some(0));
    /// let mut rng = GeneratorSeedPlan::chunk_rng(&tensor_seed, 0);
    /// let value = rng.next_f64();
    /// ```
    pub fn chunk_rng(tensor_seed: &[u8; 32], chunk_id: u64) -> DeterministicRng {
        let seed = Self::derive_chunk_seed(tensor_seed, chunk_id);
        DeterministicRng::from_seed(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_plan_creation() {
        let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        assert_eq!(plan.seed_global, 42);
        assert_eq!(plan.model_id, "glm-5.2");
        assert_eq!(plan.generation_version, "1.0.0");
    }

    #[test]
    fn seed_plan_validation() {
        let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        assert!(plan.validate().is_ok());

        let invalid_plan = GeneratorSeedPlan::new(0, "glm-5.2", "1.0.0");
        assert!(invalid_plan.validate().is_err());
    }

    #[test]
    fn tensor_seed_deterministic() {
        let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        let seed1 = plan.derive_tensor_seed("model.layers.0.mlp.gate.weight", Some(0));
        let seed2 = plan.derive_tensor_seed("model.layers.0.mlp.gate.weight", Some(0));
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn tensor_seed_depends_on_name() {
        let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        let seed1 = plan.derive_tensor_seed("tensor1", Some(0));
        let seed2 = plan.derive_tensor_seed("tensor2", Some(0));
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn tensor_seed_depends_on_layer() {
        let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        let seed1 = plan.derive_tensor_seed("tensor", Some(0));
        let seed2 = plan.derive_tensor_seed("tensor", Some(1));
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn chunk_seed_deterministic() {
        let tensor_seed = [1u8; 32];
        let seed1 = GeneratorSeedPlan::derive_chunk_seed(&tensor_seed, 0);
        let seed2 = GeneratorSeedPlan::derive_chunk_seed(&tensor_seed, 0);
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn chunk_seed_depends_on_id() {
        let tensor_seed = [1u8; 32];
        let seed1 = GeneratorSeedPlan::derive_chunk_seed(&tensor_seed, 0);
        let seed2 = GeneratorSeedPlan::derive_chunk_seed(&tensor_seed, 1);
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn rng_deterministic() {
        let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        let mut rng1 = plan.tensor_rng("tensor", Some(0));
        let mut rng2 = plan.tensor_rng("tensor", Some(0));
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn chunk_rng_deterministic() {
        let tensor_seed = [1u8; 32];
        let mut rng1 = GeneratorSeedPlan::chunk_rng(&tensor_seed, 0);
        let mut rng2 = GeneratorSeedPlan::chunk_rng(&tensor_seed, 0);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
