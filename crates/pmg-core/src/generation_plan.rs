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

//! Plan de génération déterministe (`GenerationPlan`).
//!
//! Ce module définit la structure [`GenerationPlan`] qui décrit **ce qui doit
//! être généré** sans générer immédiatement les données. Le plan est :
//! - **sérialisable** (JSON/TOML) pour inspection et rejeu ;
//! - **inspectable** : tous les champs sont publics ;
//! - **déterministe** : même plan ⇒ mêmes métadonnées de génération ;
//! - **indépendant de l'écriture disque** : aucune I/O dans ce module.
//!
//! Conformité : `docs/architecture/03-modeles-de-donnees.md` §2.4.

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{CoreError, CoreResult};
use crate::shape::Shape;

/// Plan de génération pour un tenseur unique.
///
/// Décrivant les métadonnées et paramètres de génération **sans contenir
/// les valeurs**. Le plan est la source de vérité pour le moteur de
/// génération (`pmg-generator`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationPlan {
    /// Nom complet du tenseur (ex. `"model.layers.0.mlp.gate.weight"`).
    pub tensor_name: String,
    /// Forme (dimensions strictement positives).
    pub shape: Shape,
    /// Type de données cible.
    pub dtype: DType,
    /// Seed globale du processus de génération (non nulle, politique de seed).
    pub seed: u64,
    /// Nombre d'éléments par chunk pour le découpage mémoire.
    /// Si `None`, utilise la taille par défaut du générateur.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_elements: Option<u64>,
}

impl GenerationPlan {
    /// Construit un plan de génération minimal (nom, shape, dtype, seed).
    ///
    /// # Paramètres
    /// - `tensor_name` : nom unique du tenseur ;
    /// - `shape` : forme du tenseur ;
    /// - `dtype` : type de données ;
    /// - `seed` : seed globale (non nulle).
    ///
    /// # Retourne
    /// Un plan prêt à l'emploi avec `chunk_elements = None` (défaut du générateur).
    ///
    /// # Erreurs
    /// - [`CoreError::InvalidSeed`] si `seed == 0` (politique de seed) ;
    /// - [`CoreError::InvalidShape`] si le nom est vide.
    pub fn new(
        tensor_name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        seed: u64,
    ) -> CoreResult<Self> {
        let tensor_name = tensor_name.into();
        if tensor_name.trim().is_empty() {
            return Err(CoreError::invalid_shape(
                "le nom d'un tenseur ne peut pas être vide".to_string(),
            ));
        }
        if seed == 0 {
            return Err(CoreError::InvalidSeed(
                "seed globale nulle interdite (politique de seed)".into(),
            ));
        }
        Ok(Self {
            tensor_name,
            shape,
            dtype,
            seed,
            chunk_elements: None,
        })
    }

    /// Construit un plan avec une taille de chunk explicite.
    pub fn with_chunk_elements(
        tensor_name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        seed: u64,
        chunk_elements: u64,
    ) -> CoreResult<Self> {
        let mut plan = Self::new(tensor_name, shape, dtype, seed)?;
        if chunk_elements == 0 {
            return Err(CoreError::InvalidShape(
                "chunk_elements doit être > 0".into(),
            ));
        }
        plan.chunk_elements = Some(chunk_elements);
        Ok(plan)
    }

    /// Nombre total d'éléments du tenseur (produit vérifié des dimensions).
    pub fn num_elements(&self) -> CoreResult<u64> {
        self.shape.num_elements()
    }

    /// Taille en octets du tenseur, dérivée de `shape × size_bytes(dtype)`.
    ///
    /// Retourne `None` si le dtype n'est pas émissible.
    pub fn byte_size(&self) -> CoreResult<Option<u64>> {
        match self.dtype.size_bytes() {
            Some(bytes) => {
                let n = self.num_elements()?;
                n.checked_mul(bytes).map(Some).ok_or_else(|| {
                    CoreError::Overflow(format!(
                        "taille en octets du tenseur '{}' dépasse u64::MAX",
                        self.tensor_name
                    ))
                })
            },
            None => Ok(None),
        }
    }

    /// Valide la cohérence interne du plan.
    ///
    /// Vérifie :
    /// - nom non vide ;
    /// - seed non nulle ;
    /// - dimensions strictement positives (via `Shape::new`) ;
    /// - chunk_elements > 0 si fourni.
    pub fn validate(&self) -> CoreResult<()> {
        if self.tensor_name.trim().is_empty() {
            return Err(CoreError::invalid_shape(
                "le nom d'un tenseur ne peut pas être vide".to_string(),
            ));
        }
        if self.seed == 0 {
            return Err(CoreError::InvalidSeed(
                "seed globale nulle interdite".into(),
            ));
        }
        // Reconstruit la shape pour rejouer les invariants de dimensions.
        Shape::new(self.shape.dims().to_vec())?;
        if let Some(chunk) = self.chunk_elements {
            if chunk == 0 {
                return Err(CoreError::InvalidShape(
                    "chunk_elements doit être > 0".into(),
                ));
            }
        }
        Ok(())
    }

    /// Convertit le plan en métadonnées de tenseur (sans les valeurs).
    ///
    /// Utile pour le pipeline de validation (`pmg-validate`).
    pub fn to_tensor_metadata(&self) -> CoreResult<crate::tensor_metadata::TensorMetadata> {
        crate::tensor_metadata::TensorMetadata::new(
            &self.tensor_name,
            self.shape.clone(),
            self.dtype,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::shape::Shape;

    #[test]
    fn plan_creation_minimal() {
        let shape = Shape::new(vec![64, 32]).unwrap();
        let plan = GenerationPlan::new("model.embed_tokens.weight", shape, DType::F32, 42).unwrap();
        assert_eq!(plan.tensor_name, "model.embed_tokens.weight");
        assert_eq!(plan.shape.dims(), &[64, 32]);
        assert_eq!(plan.dtype, DType::F32);
        assert_eq!(plan.seed, 42);
        assert_eq!(plan.chunk_elements, None);
    }

    #[test]
    fn plan_creation_with_chunk() {
        let shape = Shape::new(vec![100]).unwrap();
        let plan =
            GenerationPlan::with_chunk_elements("tensor", shape, DType::F16, 123, 1024).unwrap();
        assert_eq!(plan.chunk_elements, Some(1024));
    }

    #[test]
    fn plan_validation_rejects_empty_name() {
        let shape = Shape::scalar();
        assert!(GenerationPlan::new("", shape.clone(), DType::F32, 42).is_err());
        assert!(GenerationPlan::new("   ", shape, DType::F32, 42).is_err());
    }

    #[test]
    fn plan_validation_rejects_zero_seed() {
        let shape = Shape::scalar();
        assert!(GenerationPlan::new("tensor", shape, DType::F32, 0).is_err());
    }

    #[test]
    fn plan_validation_rejects_zero_chunk() {
        let shape = Shape::scalar();
        let plan = GenerationPlan::with_chunk_elements("tensor", shape, DType::F32, 42, 0);
        assert!(plan.is_err());
    }

    #[test]
    fn plan_num_elements() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let plan = GenerationPlan::new("a", shape, DType::F32, 42).unwrap();
        assert_eq!(plan.num_elements().unwrap(), 24);
    }

    #[test]
    fn plan_byte_size_f32() {
        let shape = Shape::new(vec![4, 16]).unwrap();
        let plan = GenerationPlan::new("a", shape, DType::F32, 42).unwrap();
        // 4*16*4 = 256 octets
        assert_eq!(plan.byte_size().unwrap(), Some(256));
    }

    #[test]
    fn plan_byte_size_non_emittable() {
        let shape = Shape::new(vec![4, 16]).unwrap();
        let plan = GenerationPlan::new("a", shape, DType::F4, 42).unwrap();
        assert_eq!(plan.byte_size().unwrap(), None);
    }

    #[test]
    fn plan_to_tensor_metadata() {
        let shape = Shape::new(vec![64, 32]).unwrap();
        let plan = GenerationPlan::new("model.embed_tokens.weight", shape, DType::F32, 42).unwrap();
        let meta = plan.to_tensor_metadata().unwrap();
        assert_eq!(meta.name, plan.tensor_name);
        assert_eq!(meta.shape, plan.shape);
        assert_eq!(meta.dtype, plan.dtype);
    }

    #[test]
    fn serde_roundtrip() {
        let shape = Shape::new(vec![64, 32]).unwrap();
        let plan = GenerationPlan::new("tensor", shape, DType::F32, 42).unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        let back: GenerationPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }

    #[test]
    fn serde_omits_none_chunk() {
        let shape = Shape::scalar();
        let plan = GenerationPlan::new("tensor", shape, DType::F32, 42).unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        assert!(!json.contains("chunk_elements"));
    }
}
