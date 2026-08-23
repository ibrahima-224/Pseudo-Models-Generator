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

//! Erreurs typées du générateur.

use pmg_blueprint::BlueprintError;
use pmg_core::CoreError;
use pmg_injector::InjectorError;
use pmg_math::MathError;

use crate::budget::BudgetError;

/// Erreur principale du générateur.
#[derive(Debug, thiserror::Error)]
pub enum GeneratorError {
    /// Erreur de validation de la configuration du modèle.
    #[error("erreur de configuration du modèle : {0}")]
    InvalidModelConfig(String),

    /// Erreur de blueprint (planification, validation).
    #[error("erreur de blueprint : {0}")]
    Blueprint(BlueprintError),

    /// Erreur mathématique (distribution, seed, statistiques).
    #[error("erreur mathématique : {0}")]
    Math(MathError),

    /// Erreur d'injection (pipeline structurel).
    #[error("erreur d'injection : {0}")]
    Injection(InjectorError),

    /// Erreur de validation de la génération.
    #[error("erreur de validation de génération : {0}")]
    Validation(String),

    /// Erreur de sérialisation/désérialisation.
    #[error("erreur de sérialisation : {0}")]
    Serialization(String),

    /// Erreur de déterminisme (deux générations identiques diffèrent).
    #[error("erreur de déterminisme : {0}")]
    Determinism(String),

    /// Erreur interne du générateur.
    #[error("erreur interne : {0}")]
    Internal(String),

    /// Erreur de seed (seed nulle ou invalide).
    #[error("erreur de seed : {0}")]
    SeedError(String),

    /// Erreur de chunk (taille invalide, index hors limites).
    #[error("erreur de chunk : {0}")]
    ChunkError(String),

    /// Erreur de pipeline (étape inconnue, exécution échouée).
    #[error("erreur de pipeline : {0}")]
    PipelineError(String),

    /// Erreur de tenseur (nom invalide, forme incorrecte).
    #[error("erreur de tenseur : {0}")]
    TensorError(String),

    /// Erreur de modèle (structure invalide, paramètres manquants).
    #[error("erreur de modèle : {0}")]
    ModelError(String),

    /// Erreur de budget tensoriel (budget insuffisant pour un seul élément).
    #[error("budget tensoriel insuffisant pour '{tensor_name}' : {budget} octets disponibles, {required_bytes} octets requis")]
    BudgetExceeded {
        /// Nom du tenseur.
        tensor_name: String,
        /// Budget disponible en octets.
        budget: u64,
        /// Nombre d'octets requis pour un seul élément.
        required_bytes: u64,
    },
}

/// Alias de confort pour `Result<T, GeneratorError>`.
pub type GeneratorResult<T> = Result<T, GeneratorError>;

impl From<BlueprintError> for GeneratorError {
    fn from(e: BlueprintError) -> Self {
        GeneratorError::Blueprint(e)
    }
}

impl From<MathError> for GeneratorError {
    fn from(e: MathError) -> Self {
        GeneratorError::Math(e)
    }
}

impl From<InjectorError> for GeneratorError {
    fn from(e: InjectorError) -> Self {
        GeneratorError::Injection(e)
    }
}

impl From<CoreError> for GeneratorError {
    fn from(e: CoreError) -> Self {
        GeneratorError::InvalidModelConfig(e.to_string())
    }
}

impl From<serde_json::Error> for GeneratorError {
    fn from(e: serde_json::Error) -> Self {
        GeneratorError::Serialization(e.to_string())
    }
}

impl From<BudgetError> for GeneratorError {
    fn from(e: BudgetError) -> Self {
        match e {
            BudgetError::InsufficientBudget { actual, target } => {
                GeneratorError::InvalidModelConfig(format!(
                    "PMG-204 : budget tensoriel insuffisant ({actual} < {target} octets)"
                ))
            },
            BudgetError::ToleranceExceeded {
                actual,
                target,
                tolerance,
            } => GeneratorError::Validation(format!(
                "écart hors tolérance : {actual} vs {target} (tolérance {tolerance:.2})"
            )),
            BudgetError::Internal(msg) => GeneratorError::Internal(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = GeneratorError::InvalidModelConfig("test".into());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn conversion_from_blueprint_error() {
        let bp_err = BlueprintError::PlanError("test".into());
        let gen_err: GeneratorError = bp_err.into();
        assert!(matches!(gen_err, GeneratorError::Blueprint(_)));
    }

    #[test]
    fn conversion_from_math_error() {
        let math_err = MathError::InvalidParameter("test".into());
        let gen_err: GeneratorError = math_err.into();
        assert!(matches!(gen_err, GeneratorError::Math(_)));
    }

    #[test]
    fn conversion_from_injector_error() {
        let inj_err = InjectorError::InvalidTensor("test".into());
        let gen_err: GeneratorError = inj_err.into();
        assert!(matches!(gen_err, GeneratorError::Injection(_)));
    }

    #[test]
    fn seed_error_display() {
        let err = GeneratorError::SeedError("seed nulle".into());
        assert!(err.to_string().contains("seed nulle"));
    }

    #[test]
    fn chunk_error_display() {
        let err = GeneratorError::ChunkError("index hors limites".into());
        assert!(err.to_string().contains("index hors limites"));
    }

    #[test]
    fn pipeline_error_display() {
        let err = GeneratorError::PipelineError("étape inconnue".into());
        assert!(err.to_string().contains("étape inconnue"));
    }

    #[test]
    fn tensor_error_display() {
        let err = GeneratorError::TensorError("forme invalide".into());
        assert!(err.to_string().contains("forme invalide"));
    }

    #[test]
    fn model_error_display() {
        let err = GeneratorError::ModelError("paramètres manquants".into());
        assert!(err.to_string().contains("paramètres manquants"));
    }

    #[test]
    fn budget_exceeded_display() {
        let err = GeneratorError::BudgetExceeded {
            tensor_name: "test.tensor".into(),
            budget: 3,
            required_bytes: 4,
        };
        assert!(err.to_string().contains("budget tensoriel insuffisant"));
        assert!(err.to_string().contains("test.tensor"));
    }
}
