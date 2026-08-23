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

//! Tests unitaires pour le module provenance.

use crate::provenance::granular::{FieldProvenance, GranularProvenance, TensorProvenance};
use crate::provenance::provenance_info::ProvenanceInfo;
use crate::provenance::types::{GeneratedArtifact, InputMetadata, SourceMetadata};
use pmg_core::origin::{Confidence, Origin};

#[test]
fn test_provenance_new() {
    let provenance = ProvenanceInfo::new("gen-123", 42, "size-constrained");
    assert_eq!(provenance.generation_id, "gen-123");
    assert_eq!(provenance.seed, 42);
    assert_eq!(provenance.generation_mode, "size-constrained");
    assert_eq!(provenance.schema_version, 1);
    assert!(provenance.generated_artifacts.is_empty());
}

#[test]
fn test_add_artifact() {
    let mut provenance = ProvenanceInfo::new("gen-123", 42, "full");
    let artifact = GeneratedArtifact {
        path: "output/manifest.json".to_string(),
        artifact_type: "manifest".to_string(),
        size_bytes: 1024,
        hash: "sha256:abc123".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    provenance.add_artifact(artifact);
    assert_eq!(provenance.generated_artifacts.len(), 1);
    assert_eq!(provenance.generated_artifacts[0].artifact_type, "manifest");
}

#[test]
fn test_set_input_metadata() {
    let mut provenance = ProvenanceInfo::new("gen-123", 42, "full");
    let metadata = InputMetadata {
        model_config: Some(SourceMetadata {
            path: "Models/config.json".to_string(),
            hash: "sha256:def456".to_string(),
            size_bytes: 512,
            last_modified: None,
            source_type: "config.json".to_string(),
            version: None,
        }),
        tokenizer_config: None,
        statistical_profile: None,
        blueprint: None,
        additional_sources: Vec::new(),
    };
    provenance.set_input_metadata(metadata);
    assert!(provenance.input_metadata.model_config.is_some());
    assert_eq!(
        provenance.input_metadata.model_config.unwrap().path,
        "Models/config.json"
    );
}

#[test]
fn test_validate_valid() {
    let mut provenance = ProvenanceInfo::new("gen-123", 42, "full");
    provenance.environment.start_time = "2026-01-01T00:00:00Z".to_string();
    provenance.environment.end_time = "2026-01-01T00:01:00Z".to_string();
    let artifact = GeneratedArtifact {
        path: "output/manifest.json".to_string(),
        artifact_type: "manifest".to_string(),
        size_bytes: 1024,
        hash: "sha256:abc123".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    provenance.add_artifact(artifact);
    assert!(provenance.validate().is_ok());
}

#[test]
fn test_validate_empty_generation_id() {
    let mut provenance = ProvenanceInfo::new("", 42, "full");
    let artifact = GeneratedArtifact {
        path: "output/manifest.json".to_string(),
        artifact_type: "manifest".to_string(),
        size_bytes: 1024,
        hash: "sha256:abc123".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    provenance.add_artifact(artifact);
    assert!(provenance.validate().is_err());
}

#[test]
fn test_validate_no_artifacts() {
    let provenance = ProvenanceInfo::new("gen-123", 42, "full");
    assert!(provenance.validate().is_err());
}

#[test]
fn test_validate_invalid_timestamps() {
    let mut provenance = ProvenanceInfo::new("gen-123", 42, "full");
    provenance.environment.start_time = "2026-01-01T00:01:00Z".to_string();
    provenance.environment.end_time = "2026-01-01T00:00:00Z".to_string();
    let artifact = GeneratedArtifact {
        path: "output/manifest.json".to_string(),
        artifact_type: "manifest".to_string(),
        size_bytes: 1024,
        hash: "sha256:abc123".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    provenance.add_artifact(artifact);
    assert!(provenance.validate().is_err());
}

#[test]
fn test_provenance_serialization() {
    let provenance = ProvenanceInfo::new("gen-123", 42, "full");
    let json = serde_json::to_string_pretty(&provenance).unwrap();
    let deserialized: ProvenanceInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(provenance, deserialized);
}

#[test]
fn test_provenance_display() {
    let provenance = ProvenanceInfo::new("gen-123", 42, "size-constrained");
    let display = provenance.to_string();
    assert!(display.contains("Provenance pour la génération"));
    assert!(display.contains("gen-123"));
    assert!(display.contains("size-constrained"));
}

#[test]
fn test_traceability_hash() {
    // Utiliser un timestamp fixe pour éviter les différences
    let fixed_timestamp = "2026-01-01T00:00:00Z".to_string();

    let mut provenance1 = ProvenanceInfo::new("gen-123", 42, "full");
    provenance1.timestamp_utc = fixed_timestamp.clone();

    let mut provenance2 = ProvenanceInfo::new("gen-123", 42, "full");
    provenance2.timestamp_utc = fixed_timestamp.clone();

    let mut provenance3 = ProvenanceInfo::new("gen-456", 42, "full");
    provenance3.timestamp_utc = fixed_timestamp;

    // Calculer les hashes une seule fois
    let hash1 = provenance1.traceability_hash();
    let hash2 = provenance2.traceability_hash();
    let hash3 = provenance3.traceability_hash();

    // Vérifier que les hashes sont non vides
    assert!(!hash1.is_empty());
    assert!(!hash2.is_empty());
    assert!(!hash3.is_empty());

    // Vérifier que les hashes sont les mêmes pour les mêmes données
    assert_eq!(hash1, hash2);
    // Vérifier que les hashes sont différents pour des données différentes
    assert_ne!(hash1, hash3);
}

#[test]
fn test_granular_provenance_new() {
    let gp = GranularProvenance::new();
    assert!(gp.tensor_provenance.is_empty());
    assert!(gp.field_provenance.is_empty());
    assert_eq!(gp.total_tracked(), 0);
}

#[test]
fn test_granular_provenance_validate() {
    let mut gp = GranularProvenance::new();
    gp.tensor_provenance.insert(
        "model.embed_tokens.weight".to_string(),
        TensorProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );
    gp.field_provenance.insert(
        "seed".to_string(),
        FieldProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );
    assert!(gp.validate().is_ok());
}

#[test]
fn test_granular_provenance_validate_incoherent() {
    let mut gp = GranularProvenance::new();
    gp.tensor_provenance.insert(
        "model.embed_tokens.weight".to_string(),
        TensorProvenance {
            origin: Origin::Unknown,
            confidence: Confidence::Exact,
        },
    );
    assert!(gp.validate().is_err());
}

#[test]
fn test_granular_provenance_merge() {
    let mut gp1 = GranularProvenance::new();
    gp1.tensor_provenance.insert(
        "tensor1".to_string(),
        TensorProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );

    let mut gp2 = GranularProvenance::new();
    gp2.tensor_provenance.insert(
        "tensor2".to_string(),
        TensorProvenance {
            origin: Origin::Derived,
            confidence: Confidence::Derived,
        },
    );
    gp2.field_provenance.insert(
        "field1".to_string(),
        FieldProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );

    gp1.merge(gp2);
    assert_eq!(gp1.tensor_provenance.len(), 2);
    assert_eq!(gp1.field_provenance.len(), 1);
    assert_eq!(gp1.total_tracked(), 3);
}
