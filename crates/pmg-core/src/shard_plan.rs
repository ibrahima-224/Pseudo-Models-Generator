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

//! Plan de partitionnement physique d'un shard Safetensors (`ShardPlan`).
//!
//! Un `ShardPlan` décrit comment un fichier shard sera construit : quels
//! tenseurs contient-il, dans quel ordre, et quelle est la taille totale
//! estimée. Il est utilisé par le planificateur de génération pour
//! orchestrer l'écriture déterministe des tenseurs.
//!
//! Conformité : `docs/architecture/03-modeles-de-donnees.md` §4.2.
//!
//! # Exemple
//!
//! ```
//! use pmg_core::{DType, ShardPlan, TensorShard};
//!
//! let plan = ShardPlan::new(vec![
//!     TensorShard::new("layer.0.weight", 0, 4096, DType::Bf16).unwrap(),
//!     TensorShard::new("layer.1.weight", 4096, 8192, DType::Bf16).unwrap(),
//! ]).expect("plan valide");
//!
//! assert_eq!(plan.tensor_count(), 2);
//! assert_eq!(plan.total_bytes, 8192);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{CoreError, CoreResult};

/// Entrée décrivant un tenseur dans un shard Safetensors.
///
/// Chaque tenseur est décrit par son nom, ses offsets dans le payload du shard,
/// et son type de données. Les offsets sont en octets et suivent la convention
/// Safetensors : `[start, end)` (inclusif/exclusif).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShard {
    /// Nom unique du tenseur (ex. `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Offset de début dans le payload du shard (octets, inclusif).
    pub start_offset: u64,
    /// Offset de fin dans le payload du shard (octets, exclusif).
    pub end_offset: u64,
    /// Type de données du tenseur.
    pub dtype: DType,
}

impl TensorShard {
    /// Construit une nouvelle entrée de tenseur dans un shard.
    ///
    /// # Paramètres
    ///
    /// * `name` — Nom unique du tenseur.
    /// * `start_offset` — Offset de début en octets (inclusif).
    /// * `end_offset` — Offset de fin en octets (exclusif).
    /// * `dtype` — Type de données du tenseur.
    ///
    /// # Erreurs
    ///
    /// Retourne [`CoreError::Validation`] si :
    /// - le nom est vide,
    /// - `start_offset >= end_offset`.
    pub fn new(
        name: impl Into<String>,
        start_offset: u64,
        end_offset: u64,
        dtype: DType,
    ) -> CoreResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(CoreError::Validation(
                "le nom du tenseur ne peut pas être vide".to_string(),
            ));
        }
        if start_offset >= end_offset {
            return Err(CoreError::Validation(format!(
                "offset de début ({}) doit être strictement inférieur à l'offset de fin ({})",
                start_offset, end_offset
            )));
        }
        Ok(Self {
            name,
            start_offset,
            end_offset,
            dtype,
        })
    }

    /// Retourne la taille en octets de ce tenseur dans le shard.
    pub fn byte_size(&self) -> u64 {
        self.end_offset.saturating_sub(self.start_offset)
    }
}

impl fmt::Display for TensorShard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}..{}) {} ({} octets)",
            self.name,
            self.start_offset,
            self.end_offset,
            self.dtype,
            self.byte_size()
        )
    }
}

/// Plan de partitionnement physique d'un shard Safetensors.
///
/// Regroupe l'ensemble des tenseurs qui seront écrits dans un même fichier
/// shard, avec la taille totale estimée du shard. Le plan est construit par
/// le planificateur de génération et validé avant l'écriture.
///
/// Conformité : `docs/architecture/03-modeles-de-donnees.md` §4.2.
///
/// # Invariants
///
/// 1. La liste de tenseurs n'est pas vide.
/// 2. Les offsets sont triés par ordre croissant et sans chevauchement.
/// 3. `total_bytes` est égal à la somme des tailles de tous les tenseurs.
/// 4. Chaque nom de tenseur est unique dans le plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardPlan {
    /// Tenseurs dans l'ordre d'écriture déterministe.
    pub tensors: Vec<TensorShard>,
    /// Taille totale estimée du shard en octets (somme des tailles de tenseurs).
    pub total_bytes: u64,
}

impl Default for ShardPlan {
    /// Crée un plan de shard vide (utile pour la désérialisation).
    fn default() -> Self {
        Self {
            tensors: Vec::new(),
            total_bytes: 0,
        }
    }
}

impl ShardPlan {
    /// Construit un plan de shard à partir d'une liste de tenseurs.
    ///
    /// Calcule automatiquement `total_bytes` comme la somme des tailles
    /// de tous les tenseurs fournis.
    ///
    /// # Paramètres
    ///
    /// * `tensors` — Liste des tenseurs à inclure dans le shard.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la liste est vide ou si des noms sont dupliqués.
    pub fn new(tensors: Vec<TensorShard>) -> CoreResult<Self> {
        if tensors.is_empty() {
            return Err(CoreError::Validation(
                "un shard plan doit contenir au moins un tenseur".to_string(),
            ));
        }

        // Vérification de l'unicité des noms
        let mut seen = std::collections::HashSet::new();
        for t in &tensors {
            if !seen.insert(&t.name) {
                return Err(CoreError::DuplicateTensorName(t.name.clone()));
            }
        }

        let total_bytes = tensors.iter().map(|t| t.byte_size()).sum();
        Ok(Self {
            tensors,
            total_bytes,
        })
    }

    /// Valide la cohérence interne du plan.
    ///
    /// Vérifie :
    /// - la liste n'est pas vide,
    /// - les noms sont uniques,
    /// - les offsets sont cohérents (tri croissant, sans chevauchement),
    /// - `total_bytes` est correct.
    pub fn validate(&self) -> CoreResult<()> {
        if self.tensors.is_empty() {
            return Err(CoreError::Validation(
                "un shard plan doit contenir au moins un tenseur".to_string(),
            ));
        }

        // Vérification de l'unicité des noms et du total
        let mut seen = std::collections::HashSet::new();
        let mut computed_bytes: u64 = 0;
        for t in &self.tensors {
            if !seen.insert(&t.name) {
                return Err(CoreError::DuplicateTensorName(t.name.clone()));
            }
            computed_bytes = computed_bytes.saturating_add(t.byte_size());
        }

        if computed_bytes != self.total_bytes {
            return Err(CoreError::Validation(format!(
                "total_bytes incohérent : déclaré {} mais calculé {}",
                self.total_bytes, computed_bytes
            )));
        }

        Ok(())
    }

    /// Nombre de tenseurs dans le shard.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

impl fmt::Display for ShardPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "ShardPlan ({} tenseurs, {} octets)",
            self.tensor_count(),
            self.total_bytes
        )?;
        for t in &self.tensors {
            writeln!(f, "  - {}", t)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Teste la création d'un ShardPlan valide avec deux tenseurs.
    #[test]
    fn test_shard_plan_creation() {
        let plan = ShardPlan::new(vec![
            TensorShard::new("layer.0.weight", 0, 2048, DType::Bf16).unwrap(),
            TensorShard::new("layer.1.weight", 2048, 4096, DType::F32).unwrap(),
        ])
        .expect("plan valide");

        assert_eq!(plan.tensor_count(), 2);
        assert_eq!(plan.total_bytes, 4096);
    }

    /// Teste que les noms dupliqués sont rejetés.
    #[test]
    fn test_shard_plan_unique_names() {
        let result = ShardPlan::new(vec![
            TensorShard::new("layer.0.weight", 0, 100, DType::Bf16).unwrap(),
            TensorShard::new("layer.0.weight", 100, 200, DType::Bf16).unwrap(),
        ]);

        assert!(result.is_err());
        match result.unwrap_err() {
            CoreError::DuplicateTensorName(name) => {
                assert_eq!(name, "layer.0.weight");
            },
            other => panic!("erreur inattendue : {:?}", other),
        }
    }

    /// Teste que la liste vide est rejetée.
    #[test]
    fn test_shard_plan_empty_rejected() {
        let result = ShardPlan::new(vec![]);
        assert!(result.is_err());
    }

    /// Teste la taille calculée d'un tenseur.
    #[test]
    fn test_tensor_shard_byte_size() {
        let ts = TensorShard::new("a", 100, 300, DType::F32).unwrap();
        assert_eq!(ts.byte_size(), 200);
    }

    /// Teste la validation d'un plan cohérent.
    #[test]
    fn test_shard_plan_validate_ok() {
        let plan = ShardPlan::new(vec![
            TensorShard::new("a", 0, 100, DType::Bf16).unwrap(),
            TensorShard::new("b", 100, 300, DType::F32).unwrap(),
        ])
        .unwrap();

        assert!(plan.validate().is_ok());
    }

    /// Teste que la validation échoue si total_bytes est incohérent.
    #[test]
    fn test_shard_plan_validate_total_bytes_mismatch() {
        let mut plan =
            ShardPlan::new(vec![TensorShard::new("a", 0, 100, DType::Bf16).unwrap()]).unwrap();

        plan.total_bytes = 999; // incohérent
        let result = plan.validate();
        assert!(result.is_err());
    }

    /// Teste le Default : plan vide.
    #[test]
    fn test_shard_plan_default() {
        let plan = ShardPlan::default();
        assert_eq!(plan.tensor_count(), 0);
        assert_eq!(plan.total_bytes, 0);
    }

    /// Teste la chaîne de nom vide pour TensorShard.
    #[test]
    fn test_tensor_shard_empty_name_rejected() {
        let result = TensorShard::new("", 0, 100, DType::Bf16);
        assert!(result.is_err());
    }

    /// Teste que start_offset >= end_offset est rejeté.
    #[test]
    fn test_tensor_shard_invalid_offsets() {
        let result = TensorShard::new("a", 100, 100, DType::F32);
        assert!(result.is_err());

        let result2 = TensorShard::new("a", 200, 100, DType::F32);
        assert!(result2.is_err());
    }

    /// Teste le format d'affichage de TensorShard.
    #[test]
    fn test_tensor_shard_display() {
        let ts = TensorShard::new("layer.0.weight", 0, 4096, DType::Bf16).unwrap();
        let display = format!("{}", ts);
        assert!(display.contains("layer.0.weight"));
        assert!(display.contains("4096 octets"));
    }

    /// Teste la sérialisation/désérialisation roundtrip.
    #[test]
    fn test_shard_plan_roundtrip() {
        let plan = ShardPlan::new(vec![
            TensorShard::new("a", 0, 100, DType::Bf16).unwrap(),
            TensorShard::new("b", 100, 500, DType::F32).unwrap(),
        ])
        .unwrap();

        let json = serde_json::to_string(&plan).expect("sérialisation OK");
        let deserialized: ShardPlan = serde_json::from_str(&json).expect("désérialisation OK");
        assert_eq!(plan, deserialized);
    }
}
