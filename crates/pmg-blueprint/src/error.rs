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

//! Erreurs typées de la crate `pmg-blueprint`.
//!
//! Les erreurs de description de pseudo-modèles complètent celles de
//! `pmg-core` : cohérence du blueprint, nommage, planification.

use thiserror::Error;

/// Erreur typée de la couche `pmg-blueprint`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BlueprintError {
    /// Le blueprint viole un invariant de cohérence (noms, shapes, MoE).
    #[error("blueprint invalide : {0}")]
    InvalidBlueprint(String),

    /// Le nommage généré ne correspond pas aux conventions de l'index.
    #[error("convention de nommage violée : {0}")]
    NamingError(String),

    /// La planification a produit un résultat incohérent (duplicat, ordre).
    #[error("planification invalide : {0}")]
    PlanError(String),

    /// Une erreur du noyau a été propagée.
    #[error(transparent)]
    Core(#[from] pmg_core::CoreError),
}

/// Type `Result` idiomatique de la crate `pmg-blueprint`.
pub type BlueprintResult<T> = Result<T, BlueprintError>;

#[cfg(test)]
mod tests {
    use super::BlueprintError;

    #[test]
    fn messages_are_french_and_contextual() {
        let cases: &[(BlueprintError, &str)] = &[
            (
                BlueprintError::InvalidBlueprint("noms dupliqués".into()),
                "noms dupliqués",
            ),
            (
                BlueprintError::NamingError("préfixe inconnu".into()),
                "préfixe inconnu",
            ),
            (
                BlueprintError::PlanError("ordre instable".into()),
                "ordre instable",
            ),
        ];
        for (err, expected) in cases {
            let msg = err.to_string();
            assert!(msg.contains(expected), "message '{msg}' sans '{expected}'");
        }
    }

    #[test]
    fn core_error_is_transparent() {
        // Propagation transparente d'une erreur pmg-core.
        let core = pmg_core::CoreError::Overflow("produit".into());
        let err = BlueprintError::Core(core.clone());
        assert_eq!(err.to_string(), core.to_string());
    }
}
