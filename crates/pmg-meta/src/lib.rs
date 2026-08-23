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

//! Crate `pmg-meta` — manifeste PMG, rapports et provenance.
//!
//! Sérialisation/désérialisation des artefacts de sortie du générateur :
//! `pmg_metadata.json` (schéma canonique), `pmg/statistics.json`,
//! `pmg/provenance.json`, `generation_report.json`.
//!
//! ## Responsabilité
//!
//! - schéma JSON versionné (`format_version: 1`) ;
//! - champ canonique `synthetic` et distinction
//!   OBSERVÉ/ESTIMÉ/GÉNÉRÉ/INCONNU dans la provenance ;
//! - sérialisation `serde`/`serde_json` (ajoutées au sprint 12, lot L12).
//!
//! ## Dépendances
//!
//! `pmg-core`. Interdit : I/O de poids.
//!
//! ## Modules
//!
//! - [`manifest`] : Structure `PmgMetadata` pour le manifeste canonique.
//! - [`statistics`] : Métriques agrégées de génération.
//! - [`provenance`] : Traçabilité des sources et métadonnées.
//!
//! ## Exemple
//!
//! ```rust
//! use pmg_meta::{PmgMetadata, PmgStatistics, ProvenanceInfo};
//!
//! // Création d'un manifeste
//! let metadata = PmgMetadata::new_default();
//! assert!(metadata.validate().is_ok());
//!
//! // Création de statistiques
//! let stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);
//!
//! // Création de provenance
//! let provenance = ProvenanceInfo::new("gen-123", 42, "full");
//! ```
//!
//! ## Statut
//!
//! Sprint 0 : squelette documenté, aucune API publique. Implémentation prévue
//! au sprint 12 (lot L12) et finalisée au sprint 16 (lot L16).

pub mod manifest;
pub mod provenance;
pub mod statistics;

pub use manifest::{MetadataError, PmgMetadata};
pub use provenance::ProvenanceInfo;
pub use statistics::PmgStatistics;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default() {
        let metadata = PmgMetadata::new_default();
        assert_eq!(metadata.format, "pmg-metadata");
        assert_eq!(metadata.format_version, 1);
        assert!(metadata.synthetic);
        assert!(metadata.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_format() {
        let mut metadata = PmgMetadata::new_default();
        metadata.format = "invalid".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_version() {
        let mut metadata = PmgMetadata::new_default();
        metadata.format_version = 2;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_missing_hash_prefix() {
        let mut metadata = PmgMetadata::new_default();
        metadata.source_metadata_hash = "00000000".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_zero_size() {
        let mut metadata = PmgMetadata::new_default();
        metadata.actual_size_bytes = 0;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_empty_model() {
        let mut metadata = PmgMetadata::new_default();
        metadata.model = "".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_empty_pmg_version() {
        let mut metadata = PmgMetadata::new_default();
        metadata.pmg_version = "".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_empty_generation_mode() {
        let mut metadata = PmgMetadata::new_default();
        metadata.generation_mode = "".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_empty_dtype() {
        let mut metadata = PmgMetadata::new_default();
        metadata.dtype = "".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_zero_target_size() {
        let mut metadata = PmgMetadata::new_default();
        metadata.target_size_bytes = 0;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_zero_estimated_size() {
        let mut metadata = PmgMetadata::new_default();
        metadata.estimated_size_bytes = 0;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_estimated_less_than_target() {
        let mut metadata = PmgMetadata::new_default();
        metadata.target_size_bytes = 1000;
        metadata.estimated_size_bytes = 500;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_actual_less_than_estimated() {
        let mut metadata = PmgMetadata::new_default();
        metadata.estimated_size_bytes = 1000;
        metadata.actual_size_bytes = 500;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_zero_tensor_count() {
        let mut metadata = PmgMetadata::new_default();
        metadata.tensor_count = 0;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_zero_parameter_count() {
        let mut metadata = PmgMetadata::new_default();
        metadata.parameter_count = 0;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_empty_timestamp() {
        let mut metadata = PmgMetadata::new_default();
        metadata.timestamp_utc = "".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_timestamp_format() {
        let mut metadata = PmgMetadata::new_default();
        metadata.timestamp_utc = "2026-01-01".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_timestamp_format_no_z() {
        let mut metadata = PmgMetadata::new_default();
        metadata.timestamp_utc = "2026-01-01T00:00:00".to_string();
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_json_roundtrip() {
        let metadata = PmgMetadata::new_default();
        let json = metadata.to_json().unwrap();
        let deserialized = PmgMetadata::from_json(&json).unwrap();
        assert_eq!(metadata, deserialized);
    }

    #[test]
    fn test_json_roundtrip_with_optional_fields() {
        let mut metadata = PmgMetadata::new_default();
        metadata.quantization = Some("4-bit".to_string());
        metadata.pseudo_model = Some("old-model".to_string());
        metadata.weights_are_synthetic = Some(true);
        let json = metadata.to_json().unwrap();
        let deserialized = PmgMetadata::from_json(&json).unwrap();
        assert_eq!(metadata, deserialized);
    }

    #[test]
    fn test_display_french() {
        let metadata = PmgMetadata::new_default();
        let display = metadata.display_french();
        assert!(display.contains("Manifeste PMG"));
        assert!(display.contains("glm-5.2"));
        assert!(display.contains("bf16"));
    }

    #[test]
    fn test_retrocompatibility_pseudo_model() {
        let mut metadata = PmgMetadata::new_default();
        metadata.pseudo_model = Some("old-model".to_string());
        let json = metadata.to_json().unwrap();
        assert!(json.contains("pseudo_model"));
        let deserialized = PmgMetadata::from_json(&json).unwrap();
        assert_eq!(deserialized.pseudo_model, Some("old-model".to_string()));
    }

    #[test]
    fn test_retrocompatibility_weights_are_synthetic() {
        let mut metadata = PmgMetadata::new_default();
        metadata.weights_are_synthetic = Some(true);
        let json = metadata.to_json().unwrap();
        assert!(json.contains("weights_are_synthetic"));
        let deserialized = PmgMetadata::from_json(&json).unwrap();
        assert_eq!(deserialized.weights_are_synthetic, Some(true));
    }

    #[test]
    fn test_modules_exist() {
        // Vérifie que les modules sont correctement exposés
        let _stats = PmgStatistics::new("test", "full", 123);
        let _provenance = ProvenanceInfo::new("test", 123, "full");
    }

    #[test]
    fn test_statistics_serialization() {
        let stats = PmgStatistics::new("test-model", "size-constrained", 42);
        let json = serde_json::to_string_pretty(&stats).unwrap();
        let deserialized: PmgStatistics = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_provenance_serialization() {
        let provenance = ProvenanceInfo::new("gen-123", 42, "full");
        let json = serde_json::to_string_pretty(&provenance).unwrap();
        let deserialized: ProvenanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(provenance, deserialized);
    }
}
