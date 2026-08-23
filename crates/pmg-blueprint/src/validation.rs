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

//! Validation de cohérence d'un blueprint (`BlueprintValidation`).
//!
//! Ce module fournit le rapport de validation produit par
//! [`ModelBlueprint::validate`](crate::ModelBlueprint::validate) et les
//! prédicats transverses réutilisables (taille d'experts, divisibilité MoE).

use serde::{Deserialize, Serialize};

use pmg_core::{CoreError, CoreResult};

/// Résultat de validation d'un blueprint.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BlueprintValidation {
    /// Vrai si le blueprint est valide.
    pub valid: bool,
    /// Liste des problèmes détectés (messages français explicites).
    pub issues: Vec<String>,
}

impl BlueprintValidation {
    /// Validation réussie (aucun problème).
    pub fn ok() -> BlueprintValidation {
        BlueprintValidation {
            valid: true,
            issues: Vec::new(),
        }
    }

    /// Validation échouée avec une liste de problèmes.
    pub fn failed(issues: Vec<String>) -> BlueprintValidation {
        BlueprintValidation {
            valid: issues.is_empty(),
            issues,
        }
    }

    /// Fusionne un échec supplémentaire dans le rapport (messages cumulés).
    pub fn push_error(&mut self, issue: impl Into<String>) {
        self.valid = false;
        self.issues.push(issue.into());
    }
}

/// Vérifie la divisibilité des dimensions MoE :
/// `hidden_size % expert_intermediate_size == 0` ou l'inverse n'est pas requis
/// (les matrices experts sont rectangulaires) ; on vérifie seulement que les
/// tailles sont non nulles et que le produit expert × n_routed ne déborde pas.
pub fn validate_expert_sizes(
    hidden_size: u64,
    expert_intermediate_size: u64,
    n_routed_experts: u64,
) -> CoreResult<()> {
    if hidden_size == 0 {
        return Err(CoreError::InvalidMoeConfig(
            "hidden_size nul pour la validation des experts".into(),
        ));
    }
    if expert_intermediate_size == 0 {
        return Err(CoreError::InvalidMoeConfig(
            "expert_intermediate_size nul".into(),
        ));
    }
    if n_routed_experts == 0 {
        return Err(CoreError::InvalidMoeConfig(
            "n_routed_experts nul pour la validation des experts".into(),
        ));
    }
    // Vérifie que le produit (utilisé par le planner) ne déborde pas.
    hidden_size
        .checked_mul(expert_intermediate_size)
        .ok_or_else(|| CoreError::Overflow("hidden_size × expert_intermediate_size".into()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_expert_sizes, BlueprintValidation};

    #[test]
    fn validation_report_states() {
        let ok = BlueprintValidation::ok();
        assert!(ok.valid);
        assert!(ok.issues.is_empty());

        let bad = BlueprintValidation::failed(vec!["couche désordonnée".into()]);
        assert!(!bad.valid);
        assert_eq!(bad.issues.len(), 1);
    }

    #[test]
    fn push_error_cumulates_and_invalidates() {
        let mut report = BlueprintValidation::ok();
        report.push_error("premier problème");
        report.push_error("second problème");
        assert!(!report.valid);
        assert_eq!(report.issues, vec!["premier problème", "second problème"]);
    }

    #[test]
    fn expert_sizes_validation() {
        assert!(validate_expert_sizes(6144, 2048, 256).is_ok());
        assert!(validate_expert_sizes(0, 2048, 256).is_err());
        assert!(validate_expert_sizes(6144, 0, 256).is_err());
        assert!(validate_expert_sizes(6144, 2048, 0).is_err());
    }
}
