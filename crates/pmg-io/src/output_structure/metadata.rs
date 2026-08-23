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

//! Écriture des métadonnées, statistiques et provenance.

use std::path::Path;

use pmg_core::error::CoreResult;
use pmg_core::origin::{Confidence, Origin};

use super::config::OutputConfig;
use super::utils::atomic_write;

/// Écrit le manifeste `pmg_metadata.json` dans le répertoire de sortie.
///
/// Ce fichier contient toutes les métadonnées du pseudo-modèle généré,
/// incluant le format, la version, le modèle source, le seed, les tailles,
/// et d'autres informations de traçabilité.
///
/// # Format du manifeste
///
/// ```json
/// {
///   "format": "pmg-metadata",
///   "format_version": 1,
///   "pmg_version": "1.0.0",
///   "generator_version": "1.0.0",
///   "profile_version": "glm52-v1",
///   "model": "glm-5.2",
///   "synthetic": true,
///   "seed": 42,
///   "generation_mode": "size-constrained",
///   "target_size_bytes": 1073741824,
///   "estimated_size_bytes": 1073741824,
///   "actual_size_bytes": 1074000000,
///   "tensor_count": 1240,
///   "parameter_count": 1506659919872,
///   "dtype": "bf16",
///   "quantization": null,
///   "statistical_profile": "realistic",
///   "source_metadata_hash": null,
///   "chunk_elements": 1048576,
///   "shards": 4,
///   "timestamp_utc": "2026-01-01T00:00:00Z"
/// }
/// ```
///
/// # Paramètres
/// - `output_dir` : répertoire de sortie ;
/// - `config` : configuration de la sortie.
///
/// # Erreurs
/// Retourne une erreur si l'écriture échoue.
///
/// # Exemple
///
/// ```rust,ignore
/// use pmg_io::output_structure::{write_pmg_metadata, OutputConfig, SourceModel};
/// use std::path::PathBuf;
///
/// let config = OutputConfig {
///     output_dir: PathBuf::from("/tmp/my_model"),
///     source_dir: PathBuf::from("Models/GLM-5.2"),
///     source_model: SourceModel::Glm52,
///     seed: 42,
///     generator_version: "1.0.0".to_string(),
///     timestamp_utc: "2026-01-01T00:00:00Z".to_string(),
///     parameter_count: 1506659919872,
///     tensor_count: 1240,
///     shards: 4,
///     target_size_bytes: 1073741824,
///     estimated_size_bytes: 1073741824,
///     actual_size_bytes: 1074000000,
///     dtype: "bf16".to_string(),
///     generation_mode: "size-constrained".to_string(),
/// };
///
/// write_pmg_metadata(&config.output_dir, &config).unwrap();
/// ```
pub fn write_pmg_metadata(output_dir: &Path, config: &OutputConfig) -> CoreResult<()> {
    let metadata = generate_pmg_metadata_json(config)?;
    let path = output_dir.join("pmg_metadata.json");

    atomic_write(&path, metadata.as_bytes())?;

    Ok(())
}

/// Écrit les statistiques dans pmg/statistics.json.
pub fn write_pmg_statistics(
    output_dir: &Path,
    tensors_metadata: &[pmg_core::TensorMetadata],
) -> CoreResult<()> {
    let statistics = generate_statistics_json(tensors_metadata)?;
    let path = output_dir.join("pmg").join("statistics.json");

    atomic_write(&path, statistics.as_bytes())?;

    Ok(())
}

/// Écrit la provenance dans pmg/provenance.json.
pub fn write_pmg_provenance(output_dir: &Path, config: &OutputConfig) -> CoreResult<()> {
    let provenance = generate_provenance_json(config)?;
    let path = output_dir.join("pmg").join("provenance.json");

    atomic_write(&path, provenance.as_bytes())?;

    Ok(())
}

/// Génère le contenu JSON du manifeste pmg_metadata.json.
fn generate_pmg_metadata_json(config: &OutputConfig) -> CoreResult<String> {
    let metadata = format!(
        r#"{{
  "format": "pmg-metadata",
  "format_version": 1,
  "pmg_version": "1.0.0",
  "generator_version": "{}",
  "profile_version": "{}",
  "model": "{}",
  "synthetic": true,
  "seed": {},
  "generation_mode": "{}",
  "target_size_bytes": {},
  "estimated_size_bytes": {},
  "actual_size_bytes": {},
  "tensor_count": {},
  "parameter_count": {},
  "dtype": "{}",
  "quantization": null,
  "statistical_profile": "realistic",
  "source_metadata_hash": null,
  "chunk_elements": 1048576,
  "shards": {},
  "timestamp_utc": "{}"
}}"#,
        config.generator_version,
        config.source_model.profile_version(),
        config.source_model.name(),
        config.seed,
        config.generation_mode,
        config.target_size_bytes,
        config.estimated_size_bytes,
        config.actual_size_bytes,
        config.tensor_count,
        config.parameter_count,
        config.dtype,
        config.shards,
        config.timestamp_utc
    );

    Ok(metadata)
}

/// Génère le contenu JSON des statistiques par tenseur.
///
/// Les valeurs statistiques (mean, std, outlier_rate) sont marquées comme non disponibles
/// car nous générons des données synthétiques sans informations statistiques réelles.
fn generate_statistics_json(tensors_metadata: &[pmg_core::TensorMetadata]) -> CoreResult<String> {
    let mut stats_items = Vec::new();

    for tensor in tensors_metadata {
        // Construction des statistiques avec valeurs null pour indiquer l'indisponibilité
        let stat = format!(
            r#"  {{
    "name": "{}",
    "role": "unknown",
    "distribution": "unknown",
    "mean": null,
    "std": null,
    "quantiles": {{}},
    "outlier_rate": null
  }}"#,
            tensor.name
        );
        stats_items.push(stat);
    }

    let json = format!("{{\n  \"tensors\": [\n{}\n  ]\n}}", stats_items.join(",\n"));
    Ok(json)
}

/// Génère le contenu JSON de la provenance en utilisant les enums `Origin` et `Confidence`.
///
/// Cette fonction produit un JSON structuré avec la provenance granulaire pour chaque champ
/// du modèle, en utilisant les enums définis dans `pmg-core`.
fn generate_provenance_json(_config: &OutputConfig) -> CoreResult<String> {
    // Construction de la provenance granulaire avec les enums
    let provenance = serde_json::json!({
        "model": {
            "origin": Origin::Generated,
            "confidence": Confidence::Synthetic
        },
        "seed": {
            "origin": Origin::Observed,
            "confidence": Confidence::Exact
        },
        "distribution": {
            "origin": Origin::Derived,
            "confidence": Confidence::Estimated
        },
        "structure": {
            "origin": Origin::Derived,
            "confidence": Confidence::Estimated
        },
        "outlier": {
            "origin": Origin::Derived,
            "confidence": Confidence::Estimated
        },
        "dtype": {
            "origin": Origin::Observed,
            "confidence": Confidence::Exact
        },
        "size": {
            "origin": Origin::Observed,
            "confidence": Confidence::Exact
        }
    });

    let provenance_str = serde_json::to_string_pretty(&provenance).map_err(|e| {
        pmg_core::error::CoreError::Internal(format!("Erreur de sérialisation JSON: {}", e))
    })?;

    Ok(provenance_str)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::dtype::DType;
    use pmg_core::shape::Shape;
    use pmg_core::tensor_metadata::TensorMetadata;
    use std::path::PathBuf;

    /// Test de génération du manifeste pmg_metadata.json.
    #[test]
    fn test_write_pmg_metadata() {
        let temp_dir = std::env::temp_dir().join("pmg_test_metadata");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = OutputConfig {
            output_dir: temp_dir.clone(),
            source_dir: PathBuf::from("Models/GLM-5.2"),
            source_model: super::super::config::SourceModel::Glm52,
            seed: 42,
            generator_version: "1.0.0".to_string(),
            timestamp_utc: "2026-01-01T00:00:00Z".to_string(),
            parameter_count: 1506659919872,
            tensor_count: 1240,
            shards: 4,
            target_size_bytes: 1073741824,
            estimated_size_bytes: 1073741824,
            actual_size_bytes: 1074000000,
            dtype: "bf16".to_string(),
            generation_mode: "size-constrained".to_string(),
        };

        let result = write_pmg_metadata(&temp_dir, &config);
        assert!(result.is_ok());

        // Vérifie que le fichier a été créé
        let path = temp_dir.join("pmg_metadata.json");
        assert!(path.exists());

        // Vérifie le contenu
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"synthetic\": true"));
        assert!(content.contains("\"seed\": 42"));
        assert!(content.contains("\"model\": \"glm-5.2\""));

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Test de validation des statistiques générées.
    #[test]
    fn test_validate_statistics_content() {
        let temp_dir = std::env::temp_dir().join("pmg_test_validate_stats");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Crée un répertoire pmg/
        std::fs::create_dir_all(temp_dir.join("pmg")).unwrap();

        // Crée des tenseurs de test
        let tensors = vec![
            TensorMetadata::new(
                "layer1.weight",
                Shape::new(vec![100, 100]).unwrap(),
                DType::F32,
            )
            .unwrap(),
            TensorMetadata::new("layer2.bias", Shape::new(vec![100]).unwrap(), DType::F32).unwrap(),
        ];

        let result = write_pmg_statistics(&temp_dir, &tensors);
        assert!(result.is_ok());

        // Lit et parse les statistiques
        let path = temp_dir.join("pmg").join("statistics.json");
        let content = std::fs::read_to_string(&path).unwrap();
        let stats: serde_json::Value = serde_json::from_str(&content).unwrap();

        // Vérifie la structure
        assert!(stats.is_object());
        assert!(stats.get("tensors").is_some());

        let tensors_array = stats["tensors"].as_array().unwrap();
        assert_eq!(tensors_array.len(), 2);

        // Vérifie le premier tenseur
        let tensor1 = &tensors_array[0];
        assert_eq!(tensor1["name"], "layer1.weight");
        assert_eq!(tensor1["role"], "unknown");
        assert_eq!(tensor1["distribution"], "unknown");
        assert!(tensor1["mean"].is_null());
        assert!(tensor1["std"].is_null());
        assert!(tensor1["outlier_rate"].is_null());

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
