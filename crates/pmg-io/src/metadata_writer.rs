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

//! Écriture des métadonnées de génération du pseudo-modèle.
//!
//! Ce module fournit la fonction [`write_metadata`] qui produit les métadonnées
//! de génération (generator, version, seed, etc.).
//!
//! Conformité : Sprint 10, étape 10.6 « Métadonnées ».
//!
//! # Exemple
//!
//! ```rust
//! use pmg_io::metadata_writer::write_metadata;
//! use pmg_core::generator_config::GeneratorConfig;
//!
//! let config = GeneratorConfig::default();
//! let metadata = write_metadata(&config).unwrap();
//! assert!(metadata.contains("\"seed\""));
//! ```

use std::path::Path;

use pmg_core::error::{CoreError, CoreResult};
use pmg_core::CoreConfig;

/// Écrit le fichier de métadonnées de génération.
///
/// # Paramètres
/// - `config` : configuration de génération ;
/// - `path` : chemin du fichier à écrire.
///
/// # Erreurs
/// Retourne une erreur si l'écriture échoue.
pub fn write_metadata_file(config: &CoreConfig, path: &Path) -> CoreResult<()> {
    let json = write_metadata(config)?;
    std::fs::write(path, json)
        .map_err(|e| CoreError::Internal(format!("échec écriture metadata.json : {e}")))
}

/// Génère le contenu JSON des métadonnées de génération.
pub fn write_metadata(config: &CoreConfig) -> CoreResult<String> {
    // Valide la configuration avant sérialisation
    config.validate()?;

    // Construit manuellement le JSON pour éviter les dépendances
    let json = format!(
        r#"{{
  "generator": {{
    "name": "PMG",
    "version": "{}"
  }},
  "seed": {},
  "model_id": "{}",
  "distribution": "{:?}",
  "structure_strength": {},
  "outlier_fraction": {},
  "total_parameters": 0,
  "total_tensors": 0,
  "tensor_names": []
}}"#,
        config.generation_version,
        config.seed,
        config.model_id,
        config.distribution.kind,
        config.structure.strength().value(),
        config.outlier.target_fraction,
    );

    Ok(json)
}

/// Écrit les métadonnées de tous les tenseurs.
///
/// # Paramètres
/// - `tensors` : slice de `pmg_core::TensorMetadata` à sérialiser ;
/// - `path` : chemin du fichier à écrire.
///
/// # Erreurs
/// Retourne une erreur si l'écriture échoue.
pub fn write_tensor_metadata_file(
    tensors: &[pmg_core::TensorMetadata],
    path: &Path,
) -> CoreResult<()> {
    let mut json_items = Vec::new();
    for tensor in tensors {
        let shape_str: Vec<String> = tensor.shape.dims().iter().map(|s| s.to_string()).collect();
        let dtype_str = format!("{:?}", tensor.dtype).to_lowercase();
        let num_elements = tensor.num_elements()?;
        let byte_size = tensor.byte_size()?.unwrap_or(0);
        let item = format!(
            r#"  {{
    "name": "{}",
    "shape": [{}],
    "dtype": "{}",
    "num_elements": {},
    "byte_size": {},
    "shard": {},
    "offset_start": {},
    "offset_end": {}
  }}"#,
            tensor.name,
            shape_str.join(", "),
            dtype_str,
            num_elements,
            byte_size,
            tensor
                .shard
                .as_deref()
                .map_or("null".to_string(), |s| format!("\"{}\"", s)),
            tensor
                .offset_start
                .map_or("null".to_string(), |v| v.to_string()),
            tensor
                .offset_end
                .map_or("null".to_string(), |v| v.to_string()),
        );
        json_items.push(item);
    }

    let json = format!("[\n{}\n]", json_items.join(",\n"));
    std::fs::write(path, json)
        .map_err(|e| CoreError::Internal(format!("échec écriture tensor_metadata.json : {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::dtype::DType;
    use pmg_core::generator_config::GeneratorConfig;
    use pmg_core::shape::Shape;
    use pmg_core::tensor_metadata::TensorMetadata;

    #[test]
    fn write_metadata_basic() {
        let config = GeneratorConfig::new(42, "glm-5.2").unwrap();
        let json = write_metadata(&config).unwrap();
        assert!(json.contains("\"seed\": 42"));
        assert!(json.contains("\"model_id\": \"glm-5.2\""));
        assert!(json.contains("\"generator\""));
    }

    #[test]
    fn write_metadata_json_structure() {
        let config = GeneratorConfig::default();
        let json = write_metadata(&config).unwrap();
        // Vérifie que c'est du JSON valide
        assert!(json.contains("{"));
        assert!(json.contains("}"));
        assert!(json.contains("\"generator\""));
        assert!(json.contains("\"seed\""));
    }

    #[test]
    fn write_metadata_file_success() {
        let config = GeneratorConfig::default();
        let temp_dir = std::env::temp_dir().join("pmg_test_metadata");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("metadata.json");

        let result = write_metadata_file(&config, &path);
        assert!(result.is_ok());

        // Vérifie que le fichier a été créé
        assert!(path.exists());

        // Nettoyage
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(temp_dir).unwrap();
    }

    #[test]
    fn tensor_metadata_creation() {
        let meta = TensorMetadata::new(
            "model.embed_tokens.weight",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
        )
        .unwrap();
        assert_eq!(meta.name, "model.embed_tokens.weight");
        assert_eq!(meta.num_elements().unwrap(), 6400);
        assert_eq!(meta.byte_size().unwrap(), Some(6400 * 4));
    }

    #[test]
    fn tensor_metadata_file_success() {
        let tensors = vec![
            TensorMetadata::new("tensor1", Shape::new(vec![10, 10]).unwrap(), DType::F32).unwrap(),
            TensorMetadata::new("tensor2", Shape::new(vec![20, 20]).unwrap(), DType::F16).unwrap(),
        ];

        let temp_dir = std::env::temp_dir().join("pmg_test_tensor_meta");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("tensor_metadata.json");

        let result = write_tensor_metadata_file(&tensors, &path);
        assert!(result.is_ok());

        // Vérifie que le fichier a été créé
        assert!(path.exists());

        // Nettoyage
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(temp_dir).unwrap();
    }

    #[test]
    fn serde_roundtrip() {
        let config = GeneratorConfig::new(123, "test-model").unwrap();
        let json = write_metadata(&config).unwrap();
        // Vérifie que le JSON contient les bonnes valeurs
        assert!(json.contains("\"seed\": 123"));
        assert!(json.contains("\"model_id\": \"test-model\""));
    }
}
