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

//! Erreurs typées de la crate `pmg-core`.
//!
//! Toutes les erreurs remontées par les types fondamentaux sont regroupées
//! dans [`CoreError`]. Chaque variante porte un message français explicatif et,
//! lorsque pertinent, une suggestion de correction (conformité pilier 31).

use thiserror::Error;

/// Erreur typée du noyau `pmg-core`.
///
/// Les variantes couvrent les invariants de la spécification
/// (`docs/architecture/03-modeles-de-donnees.md` §2) : dimensions, overflow,
/// dtypes non supportés, configuration incohérente, seeds invalides…
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoreError {
    /// Une dimension vaut zéro, ce que le format Safetensors interdit.
    ///
    /// Suggestion : toute dimension doit être strictement positive.
    #[error(
        "dimension nulle invalide : toutes les dimensions doivent être > 0 (format Safetensors)"
    )]
    InvalidDimension,

    /// Une shape est incohérente (rang, dimensions, produit).
    ///
    /// Suggestion : vérifier les dimensions avant de construire la shape.
    #[error("shape invalide : {0}")]
    InvalidShape(String),

    /// La configuration du modèle viole un invariant de cohérence.
    #[error("configuration de modèle invalide : {0}")]
    InvalidModelConfig(String),

    /// Le dtype demandé n'est pas supporté pour l'opération (ex. écriture v1.0).
    ///
    /// Suggestion : utiliser un dtype à taille fixe ≥ 1 octet pour l'écriture.
    #[error("dtype non supporté pour cette opération : {0}")]
    UnsupportedDType(String),

    /// Débordement arithmétique (produit de dimensions, tailles, offsets).
    ///
    /// Suggestion : utiliser des tailles réalistes ; aucune valeur n'est wrappée.
    #[error("débordement arithmétique : {0}")]
    Overflow(String),

    /// Deux tenseurs portent le même nom dans un même inventaire.
    #[error("nom de tenseur dupliqué : '{0}'")]
    DuplicateTensorName(String),

    /// Le nombre de têtes d'attention est incompatible avec les dimensions.
    #[error("têtes d'attention incompatibles : {0}")]
    IncompatibleHeads(String),

    /// La configuration MoE viole un invariant (top-k, experts, couches).
    #[error("configuration MoE invalide : {0}")]
    InvalidMoeConfig(String),

    /// La seed fournie est invalide (valeur interdite par la politique de seed).
    #[error("seed invalide : {0}")]
    InvalidSeed(String),

    /// Un champ obligatoire est absent lors de la normalisation d'une config.
    #[error("champ manquant : {0}")]
    MissingField(&'static str),

    /// Une validation de cohérence a échoué (offets, tailles, bornes).
    #[error("échec de validation : {0}")]
    Validation(String),

    /// Erreur interne inattendue (pré-condition violée dans le code).
    #[error("erreur interne : {0}")]
    Internal(String),
}

impl CoreError {
    /// Construit une erreur [`CoreError::InvalidShape`] à partir d'un message.
    pub fn invalid_shape(msg: impl Into<String>) -> Self {
        CoreError::InvalidShape(msg.into())
    }

    /// Construit une erreur [`CoreError::Overflow`] avec un contexte explicite.
    pub fn overflow(msg: impl Into<String>) -> Self {
        CoreError::Overflow(msg.into())
    }
}

/// Type `Result` idiomatique de la crate `pmg-core`.
///
/// NB : le trait `Display` est dérivé par `thiserror` à partir des attributs
/// `#[error(...)]` ; aucun impl manuel n'est nécessaire (et serait un doublon).
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::CoreError;

    #[test]
    fn display_messages_are_french_and_explicit() {
        // Chaque variante produit un message non vide, français et contextuel.
        let cases: &[(CoreError, &str)] = &[
            (CoreError::InvalidDimension, "dimension"),
            (CoreError::InvalidShape("rang 0".into()), "rang 0"),
            (
                CoreError::InvalidModelConfig("hidden_size nul".into()),
                "hidden_size nul",
            ),
            (CoreError::UnsupportedDType("F4".into()), "F4"),
            (CoreError::Overflow("produit".into()), "produit"),
            (
                CoreError::DuplicateTensorName("a.weight".into()),
                "a.weight",
            ),
            (CoreError::IncompatibleHeads("64 têtes".into()), "64 têtes"),
            (CoreError::InvalidMoeConfig("top-k".into()), "top-k"),
            (CoreError::InvalidSeed("seed 0".into()), "seed 0"),
            (CoreError::MissingField("vocab_size"), "vocab_size"),
            (
                CoreError::Validation("offset hors bornes".into()),
                "offset hors bornes",
            ),
            (
                CoreError::Internal("invariant violé".into()),
                "invariant violé",
            ),
        ];
        for (err, expected_context) in cases {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "le message ne doit pas être vide");
            // Le contexte (champ, dtype, valeur) doit apparaître dans le message.
            assert!(
                msg.contains(expected_context),
                "message '{msg}' sans le contexte attendu '{expected_context}'"
            );
        }
    }

    #[test]
    fn constructors_produce_expected_variants() {
        assert!(matches!(
            CoreError::invalid_shape("x"),
            CoreError::InvalidShape(_)
        ));
        assert!(matches!(CoreError::overflow("x"), CoreError::Overflow(_)));
    }

    #[test]
    fn errors_are_comparable_and_clonable() {
        // Utile pour les tests d'égalité de variantes.
        let a = CoreError::Overflow("m".into());
        let b = a.clone();
        assert_eq!(a, b);
    }
}
