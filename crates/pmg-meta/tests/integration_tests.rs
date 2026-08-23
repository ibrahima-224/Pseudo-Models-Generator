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

//! Tests d'intégration pour le crate pmg-meta.
//!
//! Valide la conformité complète du crate avec les spécifications.

use pmg_core::origin::{Confidence, Origin};
use pmg_meta::provenance::{FieldProvenance, GranularProvenance, TensorProvenance};
use pmg_meta::{MetadataError, PmgMetadata, PmgStatistics, ProvenanceInfo};

/// Test de validation complète du manifeste.
#[test]
fn test_manifest_validation_complete() {
    let metadata = PmgMetadata::new_default();
    assert!(
        metadata.validate().is_ok(),
        "Le manifeste par défaut doit être valide"
    );
}

/// Test de validation des champs obligatoires.
#[test]
fn test_manifest_required_fields() {
    let mut metadata = PmgMetadata::new_default();

    // Test champ model vide
    metadata.model = "".to_string();
    assert!(metadata.validate().is_err());
    assert!(matches!(
        metadata.validate().unwrap_err(),
        MetadataError::MissingField(_)
    ));

    // Restaurer pour test suivant
    metadata.model = "test".to_string();

    // Test champ pmg_version vide
    metadata.pmg_version = "".to_string();
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.pmg_version = "1.0.0".to_string();

    // Test champ generation_mode vide
    metadata.generation_mode = "".to_string();
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.generation_mode = "full".to_string();

    // Test champ dtype vide
    metadata.dtype = "".to_string();
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.dtype = "f32".to_string();
}

/// Test de validation des tailles.
#[test]
fn test_manifest_size_validation() {
    let mut metadata = PmgMetadata::new_default();

    // Test taille réelle nulle
    metadata.actual_size_bytes = 0;
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.actual_size_bytes = 1000;

    // Test taille cible nulle
    metadata.target_size_bytes = 0;
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.target_size_bytes = 1000;

    // Test taille estimée nulle
    metadata.estimated_size_bytes = 0;
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.estimated_size_bytes = 1000;

    // Test taille estimée < taille cible
    metadata.target_size_bytes = 2000;
    metadata.estimated_size_bytes = 1000;
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.target_size_bytes = 1000;
    metadata.estimated_size_bytes = 1000;

    // Test taille réelle < taille estimée
    metadata.estimated_size_bytes = 2000;
    metadata.actual_size_bytes = 1000;
    assert!(metadata.validate().is_err());
}

/// Test de validation des compteurs.
#[test]
fn test_manifest_counters_validation() {
    let mut metadata = PmgMetadata::new_default();

    // Test nombre de tenseurs nul
    metadata.tensor_count = 0;
    assert!(metadata.validate().is_err());

    // Restaurer
    metadata.tensor_count = 100;

    // Test nombre de paramètres nul
    metadata.parameter_count = 0;
    assert!(metadata.validate().is_err());
}

/// Test de validation du hash.
#[test]
fn test_manifest_hash_validation() {
    let mut metadata = PmgMetadata::new_default();

    // Test hash sans préfixe sha256:
    metadata.source_metadata_hash = "00000000".to_string();
    assert!(metadata.validate().is_err());

    // Test hash avec préfixe incorrect
    metadata.source_metadata_hash = "md5:00000000".to_string();
    assert!(metadata.validate().is_err());

    // Test hash valide
    metadata.source_metadata_hash =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_string();
    assert!(metadata.validate().is_ok());
}

/// Test de validation du timestamp.
#[test]
fn test_manifest_timestamp_validation() {
    let mut metadata = PmgMetadata::new_default();

    // Test timestamp vide
    metadata.timestamp_utc = "".to_string();
    assert!(metadata.validate().is_err());

    // Test timestamp sans T
    metadata.timestamp_utc = "2026-01-01".to_string();
    assert!(metadata.validate().is_err());

    // Test timestamp sans Z
    metadata.timestamp_utc = "2026-01-01T00:00:00".to_string();
    assert!(metadata.validate().is_err());

    // Test timestamp valide
    metadata.timestamp_utc = "2026-01-01T00:00:00Z".to_string();
    assert!(metadata.validate().is_ok());
}

/// Test de sérialisation/désérialisation du manifeste.
#[test]
fn test_manifest_serialization_roundtrip() {
    let metadata = PmgMetadata::new_default();

    // Sérialisation
    let json = metadata.to_json().unwrap();
    assert!(!json.is_empty());
    assert!(json.contains("pmg-metadata"));

    // Désérialisation
    let deserialized = PmgMetadata::from_json(&json).unwrap();
    assert_eq!(metadata, deserialized);
}

/// Test de sérialisation avec champs optionnels.
#[test]
fn test_manifest_serialization_optional_fields() {
    let mut metadata = PmgMetadata::new_default();
    metadata.quantization = Some("4-bit".to_string());
    metadata.pseudo_model = Some("old-model".to_string());
    metadata.weights_are_synthetic = Some(true);

    let json = metadata.to_json().unwrap();
    let deserialized = PmgMetadata::from_json(&json).unwrap();

    assert_eq!(metadata, deserialized);
    assert_eq!(deserialized.quantization, Some("4-bit".to_string()));
    assert_eq!(deserialized.pseudo_model, Some("old-model".to_string()));
    assert_eq!(deserialized.weights_are_synthetic, Some(true));
}

/// Test de sérialisation/désérialisation des statistiques.
#[test]
fn test_statistics_serialization_roundtrip() {
    let stats = PmgStatistics::new("test-model", "size-constrained", 42);

    let json = serde_json::to_string_pretty(&stats).unwrap();
    assert!(!json.is_empty());

    let deserialized: PmgStatistics = serde_json::from_str(&json).unwrap();
    assert_eq!(stats, deserialized);
}

/// Test de sérialisation/désérialisation de la provenance.
#[test]
fn test_provenance_serialization_roundtrip() {
    let provenance = ProvenanceInfo::new("gen-123", 42, "full");

    let json = serde_json::to_string_pretty(&provenance).unwrap();
    assert!(!json.is_empty());

    let deserialized: ProvenanceInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(provenance, deserialized);
}

/// Test de cohérence entre manifeste et statistiques.
#[test]
fn test_manifest_statistics_coherence() {
    let metadata = PmgMetadata::new_default();
    let stats = PmgStatistics::new(&metadata.model, &metadata.generation_mode, metadata.seed);

    // Vérification de la cohérence des informations de base
    assert_eq!(stats.model, metadata.model);
    assert_eq!(stats.generation_mode, metadata.generation_mode);
    assert_eq!(stats.seed, metadata.seed);
}

/// Test de cohérence entre manifeste et provenance.
#[test]
fn test_manifest_provenance_coherence() {
    let metadata = PmgMetadata::new_default();
    let provenance = ProvenanceInfo::new("test-gen", metadata.seed, &metadata.generation_mode);

    // Vérification de la cohérence des informations de base
    assert_eq!(provenance.seed, metadata.seed);
    assert_eq!(provenance.generation_mode, metadata.generation_mode);
    assert_eq!(provenance.pmg_version, metadata.pmg_version);
}

/// Test de la méthode display_french.
#[test]
fn test_manifest_display_french() {
    let metadata = PmgMetadata::new_default();
    let display = metadata.display_french();

    assert!(display.contains("Manifeste PMG"));
    assert!(display.contains(&metadata.model));
    assert!(display.contains(&metadata.dtype));
    assert!(display.contains(&metadata.generation_mode));
}

/// Test de la méthode summary des statistiques.
#[test]
fn test_statistics_summary() {
    let stats = PmgStatistics::new("test-model", "full", 123);
    let summary = stats.summary();

    assert!(summary.contains("Statistiques PMG"));
    assert!(summary.contains("test-model"));
    assert!(summary.contains("full"));
}

/// Test de la méthode summary de la provenance.
#[test]
fn test_provenance_summary() {
    let provenance = ProvenanceInfo::new("gen-123", 42, "full");
    let summary = provenance.summary();

    assert!(summary.contains("Provenance pour la génération"));
    assert!(summary.contains("gen-123"));
    assert!(summary.contains("full"));
}

/// Test de validation de la provenance.
#[test]
fn test_provenance_validation() {
    let mut provenance = ProvenanceInfo::new("gen-123", 42, "full");

    // Ajouter un artifact pour que la validation passe
    let artifact = pmg_meta::provenance::GeneratedArtifact {
        path: "output/manifest.json".to_string(),
        artifact_type: "manifest".to_string(),
        size_bytes: 1024,
        hash: "sha256:abc123".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    provenance.add_artifact(artifact);

    assert!(provenance.validate().is_ok());
}

/// Test de validation de la provenance avec erreur.
#[test]
fn test_provenance_validation_error() {
    let provenance = ProvenanceInfo::new("", 42, "full");
    assert!(provenance.validate().is_err());
}

/// Test de hash de traçabilité.
#[test]
fn test_provenance_traceability_hash() {
    // Utiliser un timestamp fixe pour éviter les différences
    let fixed_timestamp = "2026-01-01T00:00:00Z".to_string();

    let mut provenance1 = ProvenanceInfo::new("gen-123", 42, "full");
    provenance1.timestamp_utc = fixed_timestamp.clone();

    let mut provenance2 = ProvenanceInfo::new("gen-123", 42, "full");
    provenance2.timestamp_utc = fixed_timestamp.clone();

    let mut provenance3 = ProvenanceInfo::new("gen-456", 42, "full");
    provenance3.timestamp_utc = fixed_timestamp;

    assert_eq!(
        provenance1.traceability_hash(),
        provenance2.traceability_hash()
    );
    assert_ne!(
        provenance1.traceability_hash(),
        provenance3.traceability_hash()
    );
}

/// Test de sérialisation/désérialisation de GranularProvenance.
#[test]
fn test_granular_provenance_serde_roundtrip() {
    let mut gp = GranularProvenance::new();

    // Ajouter un tenseur
    gp.tensor_provenance.insert(
        "model.embed_tokens.weight".to_string(),
        TensorProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );

    // Ajouter un champ
    gp.field_provenance.insert(
        "seed".to_string(),
        FieldProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );

    // Sérialisation
    let json = serde_json::to_string_pretty(&gp).unwrap();
    assert!(json.contains("model.embed_tokens.weight"));
    assert!(json.contains("seed"));
    assert!(json.contains("Observed"));
    assert!(json.contains("Exact"));

    // Désérialisation
    let deserialized: GranularProvenance = serde_json::from_str(&json).unwrap();
    assert_eq!(gp, deserialized);
}

/// Test de cohérence des informations de provenance.
#[test]
fn test_granular_provenance_consistency() {
    let mut gp = GranularProvenance::new();

    // Ajouter des tenseurs avec différentes origines
    gp.tensor_provenance.insert(
        "tensor1".to_string(),
        TensorProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );
    gp.tensor_provenance.insert(
        "tensor2".to_string(),
        TensorProvenance {
            origin: Origin::Derived,
            confidence: Confidence::Estimated,
        },
    );
    gp.tensor_provenance.insert(
        "tensor3".to_string(),
        TensorProvenance {
            origin: Origin::Generated,
            confidence: Confidence::Synthetic,
        },
    );

    // Vérifier la validation
    assert!(gp.validate().is_ok());
    assert_eq!(gp.total_tracked(), 3);

    // Test d'incohérence (origine INCONNU avec confiance EXACT)
    let mut gp_incoherent = GranularProvenance::new();
    gp_incoherent.tensor_provenance.insert(
        "bad_tensor".to_string(),
        TensorProvenance {
            origin: Origin::Unknown,
            confidence: Confidence::Exact,
        },
    );
    assert!(gp_incoherent.validate().is_err());
}

/// Test de fusion de provenance granulaire.
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
            origin: Origin::Generated,
            confidence: Confidence::Synthetic,
        },
    );
    gp2.field_provenance.insert(
        "seed".to_string(),
        FieldProvenance {
            origin: Origin::Observed,
            confidence: Confidence::Exact,
        },
    );

    // Fusionner
    gp1.merge(gp2);

    // Vérifier que les deux tenseurs sont présents
    assert_eq!(gp1.tensor_provenance.len(), 2);
    assert!(gp1.tensor_provenance.contains_key("tensor1"));
    assert!(gp1.tensor_provenance.contains_key("tensor2"));
    assert_eq!(gp1.field_provenance.len(), 1);
    assert!(gp1.field_provenance.contains_key("seed"));
}

/// Test de provenance granulaire par défaut.
#[test]
fn test_granular_provenance_default() {
    let gp = GranularProvenance::default();
    assert!(gp.tensor_provenance.is_empty());
    assert!(gp.field_provenance.is_empty());
    assert!(gp.validate().is_ok());
    assert_eq!(gp.total_tracked(), 0);
}
