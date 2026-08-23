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

//! Erreurs typées de la crate `pmg-injector`.
//!
//! Toutes les erreurs remontées par le moteur d'injection structurelle sont
//! regroupées dans [`InjectorError`]. Chaque variante porte un message français
//! explicatif et, lorsque pertinent, une suggestion de correction (conformité
//! pilier 31 — mêmes conventions que `pmg-core::error::CoreError`).

use thiserror::Error;

/// Erreur typée du moteur d'injection `pmg-injector`.
///
/// Les variantes couvrent les invariants de la spécification
/// (`docs/architecture/04-moteurs-math-injection-generation.md` §5) :
/// paramètres de politique invalides, dimensions de tenseur incohérentes,
/// échec de validation statistique, propagation des erreurs des crates
/// `pmg-math`, `pmg-blueprint` et `pmg-core`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InjectorError {
    /// Un paramètre de la politique d'injection viole ses bornes documentées
    /// (probabilité hors `[0, 1]`, amplitude négative, rang nul…).
    #[error("paramètre de politique invalide : {0} — vérifier les bornes documentées")]
    InvalidPolicy(String),

    /// Les dimensions du tenseur sont incohérentes avec l'opération demandée
    /// (matrice attendue pour la corrélation/bas-rang, buffers de longueur
    /// différente, shape vide non admissible…).
    #[error("tenseur incohérent : {0}")]
    InvalidTensor(String),

    /// La validation statistique de l'injection a échoué : une métrique
    /// mesurée s'écarte de la cible au-delà de la tolérance demandée.
    #[error("validation d'injection échouée : {0}")]
    ValidationFailed(String),

    /// Une erreur du moteur mathématique a été propagée (paramètres de
    /// distribution, PSD, rang, données vides…).
    #[error(transparent)]
    Math(#[from] pmg_math::error::MathError),

    /// Une erreur du noyau a été propagée (shape, dtype, overflow…).
    #[error(transparent)]
    Core(#[from] pmg_core::CoreError),

    /// Une erreur de blueprint a été propagée (spécification invalide).
    #[error(transparent)]
    Blueprint(#[from] pmg_blueprint::error::BlueprintError),
}

/// Type `Result` idiomatique de la crate `pmg-injector`.
pub type InjectorResult<T> = Result<T, InjectorError>;

#[cfg(test)]
mod tests {
    use super::InjectorError;

    #[test]
    fn display_messages_are_french_and_explicit() {
        // Chaque variante produit un message non vide, français et contextuel.
        let cases: &[(InjectorError, &str)] = &[
            (
                InjectorError::InvalidPolicy("probabilité 1.5".into()),
                "probabilité 1.5",
            ),
            (
                InjectorError::InvalidTensor("shape vide".into()),
                "shape vide",
            ),
            (
                InjectorError::ValidationFailed("|p̂ − p| = 0.05".into()),
                "|p̂ − p| = 0.05",
            ),
        ];
        for (err, expected_context) in cases {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "le message ne doit pas être vide");
            assert!(
                msg.contains(expected_context),
                "message '{msg}' sans le contexte attendu '{expected_context}'"
            );
        }
    }

    #[test]
    fn errors_are_comparable_and_clonable() {
        // Utile pour les tests d'égalité de variantes.
        let a = InjectorError::InvalidPolicy("x".into());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn math_error_is_transparent() {
        // Propagation transparente d'une erreur pmg-math.
        let math = pmg_math::error::MathError::EmptyData("slice vide".into());
        let err: InjectorError = math.into();
        assert!(err.to_string().contains("slice vide"));
    }

    #[test]
    fn core_error_is_transparent() {
        // Propagation transparente d'une erreur pmg-core.
        let core = pmg_core::CoreError::Overflow("produit".into());
        let err: InjectorError = core.into();
        assert!(err.to_string().contains("produit"));
    }
}
