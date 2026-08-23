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

//! Blueprint d'un pseudo-modèle (`ModelBlueprint`) — description sans poids.
//!
//! Le blueprint est la « recette » complète : architecture, config normalisée,
//! tenseurs par catégorie (embeddings, couches, norme finale, LM head, extras)
//! et règles de nommage. Il ne contient **aucune donnée numérique**
//! (Zero-Payload, spécification §3.1).

use serde::{Deserialize, Serialize};

use pmg_core::ModelConfig;

use crate::architecture::ArchitectureKind;
use crate::error::{BlueprintError, BlueprintResult};
use crate::layer::LayerSpec;
use crate::naming::NamingRules;
use crate::tensor_spec::TensorSpec;

/// Blueprint complet d'un pseudo-modèle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelBlueprint {
    /// Identifiant du modèle (`glm-5.2`, `deepseek-v4-flash`).
    pub id: String,
    /// Famille architecturale.
    pub architecture: ArchitectureKind,
    /// Configuration normalisée (pmg-core).
    pub config: ModelConfig,
    /// Tenseurs d'embeddings (généralement 1).
    pub embeddings: Vec<TensorSpec>,
    /// Couches du transformeur (dans l'ordre).
    pub layers: Vec<LayerSpec>,
    /// Normalisation finale (`model.norm.weight` / `norm.weight`).
    pub final_norm: Vec<TensorSpec>,
    /// Tête de langage (`lm_head.weight` / `head.weight`).
    pub lm_head: Vec<TensorSpec>,
    /// Tenseurs supplémentaires (MTP, hyper-connections, indexeur…).
    pub extra_tensors: Vec<TensorSpec>,
    /// Conventions de nommage alignées sur l'index source.
    pub naming_rules: NamingRules,
    /// Résultat de la dernière validation (`validate()`), conforme au §3.1.
    #[serde(default)]
    pub validation: crate::validation::BlueprintValidation,
}

impl ModelBlueprint {
    /// Construit un blueprint vide (id + architecture + config + règles).
    pub fn new(
        id: impl Into<String>,
        architecture: ArchitectureKind,
        config: ModelConfig,
        naming_rules: NamingRules,
    ) -> ModelBlueprint {
        ModelBlueprint {
            id: id.into(),
            architecture,
            config,
            embeddings: Vec::new(),
            layers: Vec::new(),
            final_norm: Vec::new(),
            lm_head: Vec::new(),
            extra_tensors: Vec::new(),
            naming_rules,
            validation: crate::validation::BlueprintValidation::ok(),
        }
    }

    /// Tous les tenseurs du blueprint dans un ordre stable :
    /// embeddings → couches (attention, MLP, normes, hc, MoE) → norme finale
    /// → LM head → extras.
    pub fn all_tensors(&self) -> Vec<&TensorSpec> {
        let mut out = Vec::new();
        out.extend(self.embeddings.iter());
        for layer in &self.layers {
            out.extend(layer.all_tensors());
        }
        out.extend(self.final_norm.iter());
        out.extend(self.lm_head.iter());
        out.extend(self.extra_tensors.iter());
        out
    }

    /// Nombre total de tenseurs planifiés.
    pub fn tensor_count(&self) -> usize {
        self.all_tensors().len()
    }

    /// Nombre total de paramètres (Σ éléments), vérifié.
    pub fn parameter_count(&self) -> BlueprintResult<u64> {
        self.all_tensors().iter().try_fold(0u64, |acc, spec| {
            let n = spec.num_elements()?;
            acc.checked_add(n).ok_or_else(|| {
                BlueprintError::PlanError("dépassement u64 du nombre de paramètres".into())
            })
        })
    }

    /// Valide la cohérence globale : couches ordonnées, noms uniques,
    /// cohérence des couches, nombre de couches conforme.
    ///
    /// Méthode pure : retourne la première erreur sans modifier le blueprint.
    /// Utiliser [`ModelBlueprint::validate_and_report`] pour remplir le champ
    /// [`validation`](ModelBlueprint::validation) (§3.1 de la spécification).
    pub fn validate(&self) -> BlueprintResult<()> {
        // 1. Config valide (pmg-core).
        self.config.validate()?;

        // 2. Couches indexées de 0 à n-1 dans l'ordre.
        for (pos, layer) in self.layers.iter().enumerate() {
            if layer.index != pos as u64 {
                return Err(BlueprintError::InvalidBlueprint(format!(
                    "couche d'index {} à la position {pos} (les couches doivent être ordonnées)",
                    layer.index
                )));
            }
            layer.validate()?;
        }

        // 3. Nombre de couches conforme à la config.
        if self.layers.len() as u64 != self.config.num_layers {
            return Err(BlueprintError::InvalidBlueprint(format!(
                "le blueprint déclare {} couches mais la config en exige {}",
                self.layers.len(),
                self.config.num_layers
            )));
        }

        // 4. Noms uniques dans tout le blueprint.
        self.check_unique_names()?;

        Ok(())
    }

    /// Vérifie l'unicité des noms de tenseurs (BTreeSet trié → déterministe).
    fn check_unique_names(&self) -> BlueprintResult<()> {
        let mut seen = std::collections::BTreeSet::new();
        for spec in self.all_tensors() {
            if !seen.insert(spec.name.as_str()) {
                return Err(BlueprintError::InvalidBlueprint(format!(
                    "nom de tenseur dupliqué : '{}'",
                    spec.name
                )));
            }
        }
        Ok(())
    }

    /// Valide le blueprint **et** enregistre le rapport dans le champ
    /// [`validation`](ModelBlueprint::validation) (conforme §3.1).
    ///
    /// Le rapport est réinitialisé à `ok()` avant chaque exécution ; en cas
    /// d'échec, le premier problème est cumulé puis l'erreur est propagée.
    pub fn validate_and_report(&mut self) -> BlueprintResult<()> {
        self.validation = crate::validation::BlueprintValidation::ok();
        if let Err(err) = self.validate() {
            self.validation.push_error(err.to_string());
            return Err(err);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pmg_core::model_config::glm52_test_config;
    use pmg_core::{DType, Shape, TensorRole};

    use super::ModelBlueprint;
    use crate::architecture::ArchitectureKind;
    use crate::error::BlueprintError;
    use crate::layer::{LayerKind, LayerSpec};
    use crate::naming::NamingRules;
    use crate::tensor_spec::TensorSpec;

    fn tiny_blueprint(num_layers: u64) -> ModelBlueprint {
        let mut cfg = glm52_test_config();
        cfg.num_layers = num_layers;
        let mut bp = ModelBlueprint::new(
            "tiny-glm",
            ArchitectureKind::MoETransformer,
            cfg,
            NamingRules::glm52(),
        );
        bp.embeddings.push(
            TensorSpec::new(
                "model.embed_tokens.weight",
                Shape::new(vec![154880, 6144]).unwrap(),
                DType::Bf16,
                TensorRole::Embedding,
            )
            .unwrap(),
        );
        bp.final_norm.push(
            TensorSpec::new(
                "model.norm.weight",
                Shape::new(vec![6144]).unwrap(),
                DType::Bf16,
                TensorRole::Norm,
            )
            .unwrap(),
        );
        for i in 0..num_layers {
            bp.layers.push(LayerSpec::new(i, LayerKind::Dense));
        }
        bp
    }

    #[test]
    fn valid_blueprint_passes_validation() {
        let bp = tiny_blueprint(2);
        assert!(bp.validate().is_ok());
        // 1 embedding + 1 norme finale + 2 couches vides (aucun tenseur) = 2.
        assert_eq!(bp.tensor_count(), 2);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut bp = tiny_blueprint(1);
        bp.lm_head.push(
            TensorSpec::new(
                "model.embed_tokens.weight", // duplicat volontaire
                Shape::new(vec![154880, 6144]).unwrap(),
                DType::Bf16,
                TensorRole::LmHead,
            )
            .unwrap(),
        );
        let err = bp.validate().unwrap_err();
        assert!(
            matches!(err, BlueprintError::InvalidBlueprint(_)),
            "obtenu {err}"
        );
    }

    #[test]
    fn layer_count_mismatch_is_rejected() {
        let bp = tiny_blueprint(2); // 2 couches mais config 78 → échec ?
                                    // tiny_blueprint met à jour num_layers à 2 : donc valide.
        assert!(bp.validate().is_ok());
        // On désynchronise volontairement la config.
        let mut bad = bp;
        bad.config.num_layers = 3;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_and_report_updates_validation_field() {
        // Cas valide : le rapport est rempli avec valid = true.
        let mut bp = tiny_blueprint(1);
        bp.validate_and_report().unwrap();
        assert!(bp.validation.valid);
        assert!(bp.validation.issues.is_empty());

        // Cas invalide : le rapport capture le problème et l'erreur est propagée.
        let mut bad = tiny_blueprint(1);
        bad.config.num_layers = 5; // désynchronisation volontaire
        let err = bad.validate_and_report().unwrap_err();
        assert!(!bad.validation.valid);
        assert_eq!(bad.validation.issues.len(), 1);
        assert!(bad.validation.issues[0].contains(&err.to_string()));
    }

    #[test]
    fn out_of_order_layers_are_rejected() {
        let mut bp = tiny_blueprint(2);
        bp.layers[0].index = 5; // désordonne
        assert!(bp.validate().is_err());
    }

    #[test]
    fn parameter_count_is_summed() {
        let bp = tiny_blueprint(1);
        // embed [154880, 6144] + norm [6144] = 951 500 544 + 6144.
        assert_eq!(bp.parameter_count().unwrap(), 154880 * 6144 + 6144);
    }

    #[test]
    fn serde_roundtrip() {
        let bp = tiny_blueprint(1);
        let json = serde_json::to_string(&bp).unwrap();
        assert_eq!(serde_json::from_str::<ModelBlueprint>(&json).unwrap(), bp);
    }
}
