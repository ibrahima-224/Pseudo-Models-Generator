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

//! Erreurs typées de la crate `pmg-math`.
//!
//! Toutes les erreurs remontées par le moteur mathématique sont regroupées
//! dans [`MathError`]. Chaque variante porte un message français explicatif et,
//! lorsque pertinent, une suggestion de correction (conformité pilier 31).

use thiserror::Error;

/// Erreur typée du moteur mathématique `pmg-math`.
///
/// Les variantes couvrent les invariants numériques de la spécification
/// (`docs/architecture/04-moteurs-math-injection-generation.md`) : paramètres
/// invalides, matrices non semi-définies positives, rangs incohérents,
/// mélanges mal pondérés, données vides…
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MathError {
    /// Un paramètre de distribution ou d'algorithme est invalide
    /// (σ ≤ 0, df ≤ 0, poids hors bornes…).
    #[error("paramètre invalide : {0} — vérifier les bornes de validité documentées")]
    InvalidParameter(String),

    /// La matrice de covariance n'est pas semi-définie positive (PSD) :
    /// la factorisation de Cholesky est impossible ou la symétrie est violée.
    #[error("matrice non semi-définie positive : {0} — aucune correction silencieuse")]
    NotPsd(String),

    /// Un rang de décomposition bas-rang est incohérent avec les dimensions.
    #[error("rang invalide : {0} — attendu 1 ≤ rank ≤ min(m, n)")]
    InvalidRank(String),

    /// Les poids d'un mélange de distributions ne sont pas valides
    /// (négatifs, ou somme ≠ 1 à la tolérance documentée).
    #[error("poids de mélange invalides : {0}")]
    InvalidMixtureWeights(String),

    /// L'opération exige des données non vides (statistiques, quantiles…).
    #[error("données vides : {0}")]
    EmptyData(String),

    /// Une pré-condition interne a été violée (bug, jamais une entrée utilisateur).
    #[error("erreur interne : {0}")]
    Internal(String),
}

/// Type `Result` idiomatique de la crate `pmg-math`.
pub type MathResult<T> = Result<T, MathError>;

#[cfg(test)]
mod tests {
    use super::MathError;

    #[test]
    fn display_messages_are_french_and_explicit() {
        // Chaque variante produit un message non vide, français et contextuel.
        let cases: &[(MathError, &str)] = &[
            (MathError::InvalidParameter("sigma <= 0".into()), "sigma"),
            (
                MathError::NotPsd("valeur propre negative".into()),
                "valeur propre",
            ),
            (MathError::InvalidRank("rank 0".into()), "rank 0"),
            (
                MathError::InvalidMixtureWeights("somme 1.2".into()),
                "somme 1.2",
            ),
            (MathError::EmptyData("slice vide".into()), "slice vide"),
            (
                MathError::Internal("invariant viole".into()),
                "invariant viole",
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
        let a = MathError::InvalidParameter("x".into());
        let b = a.clone();
        assert_eq!(a, b);
    }
}
