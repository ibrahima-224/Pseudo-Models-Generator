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

//! Planner de budget D8 pour la génération tensorielle.
//!
//! Le planner calcule le budget tensoriel disponible selon la formule :
//! W = B − H − M − marge
//! où :
//! - B = budget total (octets)
//! - H = estimation des en-têtes (headers safetensors)
//! - M = estimation des métadonnées (config, tokenizer, etc.)
//! - marge = pourcentage de sécurité (défaut 2%)
//!
//! # Modes de génération
//!
//! - `full-structural` : toutes les shapes/dtypes du profil source ; taille théorique.
//!   Échec si budget insuffisant (PMG-204).
//! - `size-constrained` : budget D8 respecté ; choix dtype/quantification/réduction documentés.
//! - `dtype-constrained` : dtype imposé ; taille dérivée.
//!
//! # Exemple
//!
//! ```
//! use pmg_generator::budget::{BudgetPlanner, GenerationMode};
//! use pmg_blueprint::Plan;
//! use pmg_core::model_config::glm52_test_config;
//!
//! // Création d'un planner avec marge 2%
//! let planner = BudgetPlanner::new(0.02);
//!
//! // Estimation à partir d'un plan et d'une config
//! let header_est = planner.estimate_headers(&Plan { tensors: vec![], parameter_count: 0 });
//! let config = glm52_test_config();
//! let metadata_est = planner.estimate_metadata(&config);
//!
//! // Calcul du budget tensoriel
//! let budget_total = 1_000_000; // 1 Mo
//! let tensor_budget = planner.calculate_budget(budget_total, header_est, metadata_est);
//!
//! // Validation
//! assert!(planner.validate_budget(tensor_budget, tensor_budget, 0.0).is_ok());
//! ```

use thiserror::Error;

use pmg_blueprint::Plan;
use pmg_core::ModelConfig;

/// Erreur de planification budgétaire.
#[derive(Debug, Error)]
pub enum BudgetError {
    /// Budget tensoriel insuffisant en mode full-structural.
    #[error("PMG-204 : budget tensoriel insuffisant ({actual} < {target} octets)")]
    InsufficientBudget {
        /// Budget actuel calculé.
        actual: u64,
        /// Budget cible requis.
        target: u64,
    },

    /// Écart hors tolérance lors de la validation.
    #[error("écart hors tolérance : {actual} vs {target} (tolérance {tolerance:.2})")]
    ToleranceExceeded {
        /// Valeur actuelle.
        actual: u64,
        /// Valeur cible.
        target: u64,
        /// Tolérance utilisée.
        tolerance: f64,
    },

    /// Erreur interne de calcul.
    #[error("erreur interne de budget : {0}")]
    Internal(String),
}

/// Modes de génération supportés.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationMode {
    /// Mode structurel complet : shapes/dtypes du profil source.
    FullStructural,
    /// Mode contraint par la taille : budget D8 respecté.
    SizeConstrained,
    /// Mode contraint par le dtype : dtype imposé, taille dérivée.
    DtypeConstrained,
}

/// Planner de budget D8 pour la génération tensorielle.
///
/// Calcule le budget tensoriel disponible en soustrayant les en-têtes,
/// les métadonnées et une marge de sécurité du budget total.
#[derive(Debug, Clone)]
pub struct BudgetPlanner {
    /// Marge de sécurité (fraction, ex. 0.02 pour 2%).
    margin: f64,
}

impl BudgetPlanner {
    /// Crée un nouveau planner avec la marge spécifiée.
    ///
    /// # Arguments
    ///
    /// * `margin` - Marge de sécurité en fraction (0.0 à 1.0).
    ///
    /// # Cas limites
    ///
    /// - `margin < 0.0` : Panique avec message d'erreur (marge négative invalide).
    /// - `margin > 1.0` : Panique avec message d'erreur (marge supérieure à 100% invalide).
    /// - `margin = 0.0` : Aucune marge de sécurité, budget complet utilisé.
    /// - `margin = 1.0` : Marge de 100%, aucun budget tensoriel disponible.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::budget::BudgetPlanner;
    ///
    /// let planner = BudgetPlanner::new(0.02); // 2%
    /// ```
    pub fn new(margin: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&margin),
            "La marge doit être entre 0.0 et 1.0"
        );
        Self { margin }
    }

    /// Retourne la marge de sécurité actuelle.
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// Calcule le budget tensoriel disponible.
    ///
    /// Formule : W = B − H − M − marge
    ///
    /// # Arguments
    ///
    /// * `total_budget` - Budget total en octets (B).
    /// * `header_estimate` - Estimation des en-têtes en octets (H).
    /// * `metadata_estimate` - Estimation des métadonnées en octets (M).
    ///
    /// # Retour
    ///
    /// Le budget tensoriel disponible en octets (W).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::budget::BudgetPlanner;
    ///
    /// let planner = BudgetPlanner::new(0.02);
    /// let budget = planner.calculate_budget(1_000_000, 10_000, 50_000);
    /// // budget ≈ 1_000_000 − 10_000 − 50_000 − 20_000 = 920_000
    /// ```
    pub fn calculate_budget(
        &self,
        total_budget: u64,
        header_estimate: u64,
        metadata_estimate: u64,
    ) -> u64 {
        // Calcul de la marge en octets
        let margin_bytes = (total_budget as f64 * self.margin) as u64;

        // W = B − H − M − marge
        total_budget
            .saturating_sub(header_estimate)
            .saturating_sub(metadata_estimate)
            .saturating_sub(margin_bytes)
    }

    /// Estime la taille des en-têtes à partir d'un plan.
    ///
    /// Chaque tenseur nécessite un en-tête safetensors contenant :
    /// - Nom du tenseur (longueur variable, estimée à 64 octets en moyenne)
    /// - Offset et taille (16 octets)
    /// - Aligne sur 8 octets
    ///
    /// # Arguments
    ///
    /// * `plan` - Plan de génération contenant la liste des tenseurs.
    ///
    /// # Retour
    ///
    /// Estimation de la taille totale des en-têtes en octets.
    pub fn estimate_headers(&self, plan: &Plan) -> u64 {
        // Estimation par tenseur : 64 (nom) + 16 (offset/taille) = 80 octets
        // Alignement moyen de 8 octets → arrondi à 88 octets par tenseur
        let per_tensor_header = 88u64;
        let tensor_count = plan.tensors.len() as u64;

        // En-tête global safetensors : 8 octets (magic) + 8 octets (nombre de tenseurs)
        let global_header = 16u64;

        global_header + tensor_count * per_tensor_header
    }

    /// Estime la taille des métadonnées à partir de la configuration du modèle.
    ///
    /// Inclut : config.json, tokenizer.json, generation_config.json,
    /// pmg_metadata.json, provenance.json, statistics.json.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration du modèle source.
    ///
    /// # Retour
    ///
    /// Estimation de la taille totale des métadonnées en octets.
    pub fn estimate_metadata(&self, _config: &ModelConfig) -> u64 {
        // Estimation conservatrice basée sur des modèles typiques :
        // - config.json : ~2 Ko
        // - tokenizer.json : ~10 Ko
        // - generation_config.json : ~1 Ko
        // - pmg_metadata.json : ~5 Ko
        // - provenance.json : ~2 Ko
        // - statistics.json : ~3 Ko
        // Total : ~23 Ko → arrondi à 24 Ko
        24 * 1024
    }

    /// Valide le budget actuel par rapport à un cible avec une tolérance donnée.
    ///
    /// # Arguments
    ///
    /// * `actual` - Budget actuel calculé.
    /// * `target` - Budget cible requis.
    /// * `tolerance` - Tolérance acceptée en fraction (ex. 0.02 pour 2%).
    ///
    /// # Retour
    ///
    /// `Ok(())` si la validation passe, `Err(BudgetError)` sinon.
    ///
    /// # Erreurs
    ///
    /// Retourne `BudgetError::ToleranceExceeded` si l'écart dépasse la tolérance.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::budget::BudgetPlanner;
    ///
    /// let planner = BudgetPlanner::new(0.02);
    /// assert!(planner.validate_budget(100, 100, 0.0).is_ok());
    /// assert!(planner.validate_budget(98, 100, 0.02).is_ok());
    /// assert!(planner.validate_budget(95, 100, 0.02).is_err());
    /// ```
    pub fn validate_budget(
        &self,
        actual: u64,
        target: u64,
        tolerance: f64,
    ) -> Result<(), BudgetError> {
        if target == 0 {
            return Ok(());
        }

        let diff = actual.abs_diff(target);

        let relative_diff = diff as f64 / target as f64;

        if relative_diff <= tolerance {
            Ok(())
        } else {
            Err(BudgetError::ToleranceExceeded {
                actual,
                target,
                tolerance,
            })
        }
    }

    /// Vérifie si le budget est suffisant pour le mode spécifié.
    ///
    /// En mode `full-structural`, retourne une erreur PMG-204 si le budget
    /// actuel est inférieur au budget requis.
    ///
    /// # Arguments
    ///
    /// * `mode` - Mode de génération.
    /// * `actual_budget` - Budget tensoriel calculé.
    /// * `required_budget` - Budget requis pour le modèle complet.
    ///
    /// # Retour
    ///
    /// `Ok(())` si le budget est suffisant, `Err(BudgetError::InsufficientBudget)` sinon.
    pub fn check_budget_for_mode(
        &self,
        mode: &GenerationMode,
        actual_budget: u64,
        required_budget: u64,
    ) -> Result<(), BudgetError> {
        match mode {
            GenerationMode::FullStructural => {
                if actual_budget < required_budget {
                    Err(BudgetError::InsufficientBudget {
                        actual: actual_budget,
                        target: required_budget,
                    })
                } else {
                    Ok(())
                }
            },
            GenerationMode::SizeConstrained | GenerationMode::DtypeConstrained => {
                // Ces modes acceptent un budget inférieur (réductions documentées)
                Ok(())
            },
        }
    }
}

impl Default for BudgetPlanner {
    fn default() -> Self {
        // Marge de sécurité par défaut : 2%
        Self::new(0.02)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_planner_creation() {
        let planner = BudgetPlanner::new(0.02);
        assert_eq!(planner.margin(), 0.02);
    }

    #[test]
    fn test_budget_planner_default() {
        let planner = BudgetPlanner::default();
        assert_eq!(planner.margin(), 0.02);
    }

    #[test]
    fn test_calculate_budget() {
        let planner = BudgetPlanner::new(0.02);
        let budget = planner.calculate_budget(1_000_000, 10_000, 50_000);
        // 1_000_000 − 10_000 − 50_000 − 20_000 (2% de 1M) = 920_000
        assert_eq!(budget, 920_000);
    }

    #[test]
    fn test_estimate_headers() {
        let planner = BudgetPlanner::default();
        let plan = Plan {
            tensors: vec![],
            parameter_count: 0,
        };
        let headers = planner.estimate_headers(&plan);
        // 16 (global) + 0 * 88 = 16
        assert_eq!(headers, 16);
    }

    #[test]
    fn test_estimate_metadata() {
        let planner = BudgetPlanner::default();
        let config = pmg_core::model_config::glm52_test_config();
        let metadata = planner.estimate_metadata(&config);
        // 24 Ko
        assert_eq!(metadata, 24 * 1024);
    }

    #[test]
    fn test_validate_budget_ok() {
        let planner = BudgetPlanner::default();
        assert!(planner.validate_budget(100, 100, 0.0).is_ok());
        assert!(planner.validate_budget(98, 100, 0.02).is_ok());
    }

    #[test]
    fn test_validate_budget_error() {
        let planner = BudgetPlanner::default();
        assert!(planner.validate_budget(95, 100, 0.02).is_err());
    }

    #[test]
    fn test_check_budget_for_mode_full_structural_ok() {
        let planner = BudgetPlanner::default();
        let result = planner.check_budget_for_mode(&GenerationMode::FullStructural, 1000, 900);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_budget_for_mode_full_structural_error() {
        let planner = BudgetPlanner::default();
        let result = planner.check_budget_for_mode(&GenerationMode::FullStructural, 800, 1000);
        assert!(result.is_err());
        match result {
            Err(BudgetError::InsufficientBudget { actual, target }) => {
                assert_eq!(actual, 800);
                assert_eq!(target, 1000);
            },
            _ => panic!("Erreur attendue : InsufficientBudget"),
        }
    }

    #[test]
    fn test_check_budget_for_mode_size_constrained() {
        let planner = BudgetPlanner::default();
        // Même si le budget est insuffisant, le mode size-constrained accepte
        let result = planner.check_budget_for_mode(&GenerationMode::SizeConstrained, 800, 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_budget_for_mode_dtype_constrained() {
        let planner = BudgetPlanner::default();
        let result = planner.check_budget_for_mode(&GenerationMode::DtypeConstrained, 800, 1000);
        assert!(result.is_ok());
    }
}
