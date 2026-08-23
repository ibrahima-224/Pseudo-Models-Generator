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

//! Manifeste décrivant le contenu du pseudo-modèle.
//!
//! Ce module définit la structure [`Manifest`] qui décrit le contenu complet
//! d'un pseudo-modèle : type, architecture, tenseurs, paramètres, etc.
//!
//! Conformité : Sprint 10, étape 10.7 « Manifest ».
//!
//! # Exemple
//!
//! ```rust
//! use pmg_core::manifest::{Manifest, TensorInfo};
//!
//! let mut manifest = Manifest::new("glm-5.2", "transformer");
//! manifest.add_tensor(TensorInfo::new(
//!     "model.embed_tokens.weight",
//!     vec![100, 64],
//!     "f32",
//! ));
//!
//! assert_eq!(manifest.num_tensors(), 1);
//! assert_eq!(manifest.total_parameters(), 6400);
//! ```

use crate::error::{CoreError, CoreResult};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Type de modèle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    /// Pseudo-modèle généré.
    #[serde(rename = "pseudo-modèle")]
    PseudoModel,
    /// Modèle réel (pour référence).
    #[serde(rename = "modèle réel")]
    RealModel,
}

impl ModelType {
    /// Nom lisible.
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelType::PseudoModel => "pseudo-modèle",
            ModelType::RealModel => "modèle réel",
        }
    }
}

/// Architecture du modèle.
#[derive(Debug, Clone, PartialEq)]
pub enum Architecture {
    /// Transformer standard.
    Transformer,
    /// Mixture of Experts.
    MoETransformer,
    /// Autre architecture.
    Other(String),
}

impl Serialize for Architecture {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Architecture::Transformer => serializer.serialize_str("transformer"),
            Architecture::MoETransformer => serializer.serialize_str("mixture-of-experts"),
            Architecture::Other(name) => serializer.serialize_str(name),
        }
    }
}

impl<'de> Deserialize<'de> for Architecture {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "transformer" => Ok(Architecture::Transformer),
            "mixture-of-experts" => Ok(Architecture::MoETransformer),
            _ => Ok(Architecture::Other(s)),
        }
    }
}

impl Architecture {
    /// Nom lisible.
    pub fn display_name(&self) -> String {
        match self {
            Architecture::Transformer => "transformer".to_string(),
            Architecture::MoETransformer => "mixture-of-experts".to_string(),
            Architecture::Other(name) => name.clone(),
        }
    }
}

/// Information sur un tenseur.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Nom du tenseur.
    pub name: String,
    /// Shape du tenseur (dimensions en `u64` pour cohérence avec `Shape`).
    pub shape: Vec<u64>,
    /// Type de données.
    pub dtype: String,
    /// Nombre d'éléments (calculé automatiquement).
    #[serde(skip)]
    pub num_elements: u64,
    /// Taille en octets (calculée automatiquement).
    #[serde(skip)]
    pub byte_size: u64,
}

impl TensorInfo {
    /// Crée les informations d'un tenseur.
    ///
    /// # Paramètres
    /// - `name`: Nom du tenseur (ex: "model.embed_tokens.weight").
    /// - `shape`: Dimensions du tenseur en `u64` (cohérent avec `Shape`).
    /// - `dtype`: Type de données (ex: "f32", "bf16").
    pub fn new(name: impl Into<String>, shape: Vec<u64>, dtype: impl Into<String>) -> Self {
        // Calcul du nombre total d'éléments sans débordement (shape est validée)
        let num_elements = shape.iter().product::<u64>();
        let dtype_str = dtype.into();
        let byte_size = match dtype_str.as_str() {
            "f32" => num_elements * 4,
            "f16" | "bf16" => num_elements * 2,
            "f64" => num_elements * 8,
            _ => num_elements,
        };

        Self {
            name: name.into(),
            shape,
            dtype: dtype_str,
            num_elements,
            byte_size,
        }
    }

    /// Recalcule les champs dérivés (num_elements, byte_size) à partir de shape et dtype.
    /// Utile après désérialisation.
    pub fn recompute_derived_fields(&mut self) {
        // Calcul sans débordement (shape est validée lors de la création)
        self.num_elements = self.shape.iter().product::<u64>();
        self.byte_size = match self.dtype.as_str() {
            "f32" => self.num_elements * 4,
            "f16" | "bf16" => self.num_elements * 2,
            "f64" => self.num_elements * 8,
            _ => self.num_elements,
        };
    }
}

/// Manifeste décrivant le contenu du pseudo-modèle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Type de modèle.
    pub model_type: ModelType,
    /// Nom du modèle.
    pub model_name: String,
    /// Architecture.
    pub architecture: Architecture,
    /// Version du générateur.
    pub generator_version: String,
    /// Seed de génération.
    pub seed: u64,
    /// Liste des tenseurs.
    pub tensors: Vec<TensorInfo>,
    /// Nombre total de paramètres.
    pub total_parameters: u64,
    /// Nombre total de tenseurs.
    pub total_tensors: u64,
}

impl Manifest {
    /// Crée un nouveau manifeste.
    pub fn new(model_name: impl Into<String>, architecture: impl Into<String>) -> Self {
        let arch_str = architecture.into();
        let architecture = match arch_str.as_str() {
            "transformer" => Architecture::Transformer,
            "moe-transformer" | "mixture-of-experts" => Architecture::MoETransformer,
            _ => Architecture::Other(arch_str),
        };

        Self {
            model_type: ModelType::PseudoModel,
            model_name: model_name.into(),
            architecture,
            generator_version: crate::PMG_VERSION.to_string(),
            seed: 42,
            tensors: Vec::new(),
            total_parameters: 0,
            total_tensors: 0,
        }
    }

    /// Ajoute un tenseur au manifeste.
    pub fn add_tensor(&mut self, tensor: TensorInfo) {
        self.total_parameters += tensor.num_elements;
        self.total_tensors += 1;
        self.tensors.push(tensor);
    }

    /// Retourne le nombre de tenseurs.
    pub fn num_tensors(&self) -> u64 {
        self.total_tensors
    }

    /// Retourne le nombre total de paramètres.
    pub fn total_parameters(&self) -> u64 {
        self.total_parameters
    }

    /// Retourne la taille totale en octets.
    pub fn total_byte_size(&self) -> u64 {
        self.tensors.iter().map(|t| t.byte_size).sum()
    }

    /// Valide la cohérence du manifeste.
    pub fn validate(&self) -> CoreResult<()> {
        if self.model_name.trim().is_empty() {
            return Err(CoreError::Validation(
                "model_name ne peut pas être vide".into(),
            ));
        }
        if self.seed == 0 {
            return Err(CoreError::InvalidSeed("seed ne peut pas être nulle".into()));
        }

        // Vérifie la cohérence des compteurs
        let computed_params: u64 = self.tensors.iter().map(|t| t.num_elements).sum();
        if computed_params != self.total_parameters {
            return Err(CoreError::Validation(format!(
                "total_parameters ({}) ne correspond pas à la somme des tenseurs ({})",
                self.total_parameters, computed_params
            )));
        }

        let computed_tensors = self.tensors.len() as u64;
        if computed_tensors != self.total_tensors {
            return Err(CoreError::Validation(format!(
                "total_tensors ({}) ne correspond pas au nombre réel ({})",
                self.total_tensors, computed_tensors
            )));
        }

        Ok(())
    }

    /// Sérialise le manifeste en JSON (format standard via serde).
    pub fn to_json(&self) -> CoreResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| CoreError::Internal(e.to_string()))
    }

    /// Désérialise un manifeste depuis une chaîne JSON.
    ///
    /// Recalcule automatiquement les champs dérivés (`num_elements`, `byte_size`)
    /// de chaque tenseur après désérialisation.
    pub fn from_json(json: &str) -> CoreResult<Self> {
        let mut manifest: Manifest =
            serde_json::from_str(json).map_err(|e| CoreError::Internal(e.to_string()))?;

        // Recalcule les champs dérivés pour chaque tenseur
        for tensor in &mut manifest.tensors {
            tensor.recompute_derived_fields();
        }

        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_creation() {
        let manifest = Manifest::new("glm-5.2", "transformer");
        assert_eq!(manifest.model_name, "glm-5.2");
        assert_eq!(manifest.architecture, Architecture::Transformer);
        assert_eq!(manifest.total_tensors, 0);
        assert_eq!(manifest.total_parameters, 0);
    }

    #[test]
    fn manifest_add_tensor() {
        let mut manifest = Manifest::new("test", "transformer");
        let tensor = TensorInfo::new("weight", vec![10u64, 10], "f32");
        manifest.add_tensor(tensor);

        assert_eq!(manifest.num_tensors(), 1);
        assert_eq!(manifest.total_parameters(), 100);
    }

    #[test]
    fn manifest_multiple_tensors() {
        let mut manifest = Manifest::new("test", "transformer");
        manifest.add_tensor(TensorInfo::new("w1", vec![10u64, 10], "f32"));
        manifest.add_tensor(TensorInfo::new("w2", vec![20u64, 20], "f16"));

        assert_eq!(manifest.num_tensors(), 2);
        assert_eq!(manifest.total_parameters(), 100 + 400);
        assert_eq!(manifest.total_byte_size(), 100 * 4 + 400 * 2);
    }

    #[test]
    fn manifest_validation() {
        let mut manifest = Manifest::new("test", "transformer");
        manifest.add_tensor(TensorInfo::new("w", vec![10u64, 10], "f32"));
        assert!(manifest.validate().is_ok());

        // Invalide : nom vide
        let manifest = Manifest::new("", "transformer");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn manifest_json() {
        let mut manifest = Manifest::new("test", "transformer");
        manifest.add_tensor(TensorInfo::new("w", vec![10u64, 10], "f32"));
        let json = manifest.to_json().unwrap();

        assert!(json.contains("\"model_name\": \"test\""));
        assert!(json.contains("\"total_parameters\": 100"));
        assert!(json.contains("\"w\""));
    }

    #[test]
    fn architecture_display() {
        assert_eq!(Architecture::Transformer.display_name(), "transformer");
        assert_eq!(
            Architecture::MoETransformer.display_name(),
            "mixture-of-experts"
        );
        assert_eq!(
            Architecture::Other("custom".to_string()).display_name(),
            "custom"
        );
    }

    #[test]
    fn tensor_info_creation() {
        let tensor = TensorInfo::new("test", vec![10u64, 20, 30], "f32");
        assert_eq!(tensor.name, "test");
        assert_eq!(tensor.num_elements, 6000);
        assert_eq!(tensor.byte_size, 6000 * 4);
    }

    #[test]
    fn manifest_from_json_roundtrip() {
        // Création d'un manifeste de test
        let mut manifest = Manifest::new("test", "transformer");
        manifest.seed = 123;
        manifest.add_tensor(TensorInfo::new("w1", vec![10u64, 10], "f32"));
        manifest.add_tensor(TensorInfo::new("w2", vec![20u64, 20], "f16"));

        // Sérialisation en JSON
        let json = manifest.to_json().unwrap();

        // Désérialisation depuis le JSON
        let manifest2 = Manifest::from_json(&json).unwrap();

        // Vérification de la cohérence
        assert_eq!(manifest.model_name, manifest2.model_name);
        assert_eq!(manifest.model_type, manifest2.model_type);
        assert_eq!(manifest.architecture, manifest2.architecture);
        assert_eq!(manifest.generator_version, manifest2.generator_version);
        assert_eq!(manifest.seed, manifest2.seed);
        assert_eq!(manifest.total_parameters, manifest2.total_parameters);
        assert_eq!(manifest.total_tensors, manifest2.total_tensors);
        assert_eq!(manifest.tensors.len(), manifest2.tensors.len());

        // Vérification que les champs dérivés sont recalculés correctement
        for (t1, t2) in manifest.tensors.iter().zip(manifest2.tensors.iter()) {
            assert_eq!(t1.name, t2.name);
            assert_eq!(t1.shape, t2.shape);
            assert_eq!(t1.dtype, t2.dtype);
            assert_eq!(t1.num_elements, t2.num_elements);
            assert_eq!(t1.byte_size, t2.byte_size);
        }
    }
}
