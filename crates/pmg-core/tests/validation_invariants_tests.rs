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

//! Tests de validation des invariants du crate `pmg-core`.
//!
//! Vérifie que les invariants fondamentaux sont correctement implémentés :
//! - dimensions strictement positives
//! - tailles > 0
//! - plages d'offsets valides
//! - divisibilité des têtes
//! - etc.

use pmg_core::dtype::DType;
use pmg_core::error::CoreError;
use pmg_core::model_config::{glm52_test_config, ModelConfig};
use pmg_core::moe::glm52_moe_config;
use pmg_core::shape::Shape;
use pmg_core::tensor_metadata::TensorMetadata;
use pmg_core::validation::{
    validate_at_least_one, validate_divisible_by, validate_non_zero_size, validate_offset_range,
    validate_strictly_positive,
};

/// Test des invariants de Shape : dimensions strictement positives.
#[test]
fn shape_invariants() {
    // Shape valide.
    let shape = Shape::new(vec![10, 20, 30]).unwrap();
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.num_elements().unwrap(), 6000);

    // Shape scalaire.
    let scalar = Shape::scalar();
    assert!(scalar.is_scalar());
    assert_eq!(scalar.num_elements().unwrap(), 1);

    // Dimension nulle rejetée.
    assert!(Shape::new(vec![0, 10]).is_err());
    assert!(Shape::new(vec![10, 0]).is_err());
    assert!(Shape::new(vec![10, 0, 20]).is_err());
}

/// Test des invariants de TensorMetadata : nom non vide, tailles cohérentes.
#[test]
fn tensor_metadata_invariants() {
    // Métadonnées valides.
    let meta =
        TensorMetadata::new("weight", Shape::new(vec![10, 20]).unwrap(), DType::F32).unwrap();
    assert_eq!(meta.byte_size().unwrap(), Some(800)); // 10*20*4
    assert!(meta.validate().is_ok());

    // Nom vide rejeté.
    assert!(TensorMetadata::new("", Shape::scalar(), DType::F32).is_err());
    assert!(TensorMetadata::new("   ", Shape::scalar(), DType::F32).is_err());

    // Dtype non émissible.
    let meta_q = TensorMetadata::new("q", Shape::new(vec![10, 20]).unwrap(), DType::F4).unwrap();
    assert_eq!(meta_q.byte_size().unwrap(), None);
    assert!(meta_q.validate().is_ok());
}

/// Test des invariants de ModelConfig : champs obligatoires >= 1.
#[test]
fn model_config_invariants() {
    let config = glm52_test_config();
    assert!(config.validate().is_ok());

    // Test avec num_layers = 0.
    let mut config_bad = config.clone();
    config_bad.num_layers = 0;
    assert!(config_bad.validate().is_err());

    // Test avec vocab_size = 0.
    let mut config_bad = config.clone();
    config_bad.vocab_size = 0;
    assert!(config_bad.validate().is_err());

    // Test avec hidden_size = 0.
    let mut config_bad = config.clone();
    config_bad.hidden_size = 0;
    assert!(config_bad.validate().is_err());

    // Test avec rope_theta = 0.
    let mut config_bad = config.clone();
    config_bad.rope_theta = 0.0;
    assert!(config_bad.validate().is_err());

    // Test avec rms_norm_eps = 0.
    let mut config_bad = config.clone();
    config_bad.rms_norm_eps = 0.0;
    assert!(config_bad.validate().is_err());
}

/// Test des invariants MoE : top-k <= total experts, experts >= 1.
#[test]
fn moe_invariants() {
    let moe = glm52_moe_config();
    assert!(moe.validate().is_ok());

    // Test avec n_routed_experts = 0.
    let mut moe_bad = moe.clone();
    moe_bad.n_routed_experts = 0;
    assert!(moe_bad.validate().is_err());

    // Test avec experts_per_tok = 0.
    let mut moe_bad = moe.clone();
    moe_bad.experts_per_tok = 0;
    assert!(moe_bad.validate().is_err());

    // Test avec experts_per_tok > total_experts.
    let mut moe_bad = moe.clone();
    moe_bad.experts_per_tok = 1000;
    assert!(moe_bad.validate().is_err());

    // Test avec routed_scaling_factor = 0.
    let mut moe_bad = moe.clone();
    moe_bad.routed_scaling_factor = 0.0;
    assert!(moe_bad.validate().is_err());

    // Test avec routed_scaling_factor négatif.
    let mut moe_bad = moe.clone();
    moe_bad.routed_scaling_factor = -1.0;
    assert!(moe_bad.validate().is_err());
}

/// Test des fonctions de validation partagées.
#[test]
fn shared_validation_functions() {
    // validate_non_zero_size.
    assert!(validate_non_zero_size(1, "hidden_size").is_ok());
    assert!(validate_non_zero_size(0, "hidden_size").is_err());

    // validate_at_least_one.
    assert!(validate_at_least_one(78, "num_layers").is_ok());
    assert!(validate_at_least_one(0, "num_layers").is_err());

    // validate_strictly_positive.
    assert!(validate_strictly_positive(1e-5, "rms_norm_eps").is_ok());
    assert!(validate_strictly_positive(0.0, "rms_norm_eps").is_err());
    assert!(validate_strictly_positive(f64::NAN, "x").is_err());
    assert!(validate_strictly_positive(f64::INFINITY, "x").is_err());

    // validate_divisible_by.
    assert_eq!(validate_divisible_by(6144, 64, "GLM").unwrap(), 96);
    assert!(validate_divisible_by(6144, 0, "x").is_err());
    assert!(validate_divisible_by(10, 3, "x").is_err());

    // validate_offset_range.
    assert!(validate_offset_range(0, 8, 8, false).is_ok());
    assert!(validate_offset_range(8, 4, 4, false).is_err());
    assert!(validate_offset_range(0, 10, 8, false).is_err());
    assert!(validate_offset_range(4, 4, 0, true).is_ok());
    assert!(validate_offset_range(4, 4, 0, false).is_err());
}

/// Test de cohérence entre Shape et TensorMetadata.
#[test]
fn shape_tensor_consistency() {
    let shape = Shape::new(vec![10, 20, 30]).unwrap();
    let meta = TensorMetadata::new("tensor", shape.clone(), DType::F32).unwrap();

    // Vérifie que la shape est correctement stockée.
    assert_eq!(meta.shape, shape);
    assert_eq!(meta.num_elements().unwrap(), shape.num_elements().unwrap());

    // Vérifie la taille en octets.
    let expected_bytes = 10 * 20 * 30 * 4; // F32 = 4 octets
    assert_eq!(meta.byte_size().unwrap(), Some(expected_bytes));
}

/// Test de sérialisation/désérialisation (roundtrip).
#[test]
fn serialization_roundtrip() {
    let shape = Shape::new(vec![4, 16]).unwrap();
    let json = serde_json::to_string(&shape).unwrap();
    assert_eq!(serde_json::from_str::<Shape>(&json).unwrap(), shape);

    let meta = TensorMetadata::new("w", Shape::new(vec![2, 2]).unwrap(), DType::Bf16).unwrap();
    let json = serde_json::to_string(&meta).unwrap();
    assert_eq!(serde_json::from_str::<TensorMetadata>(&json).unwrap(), meta);

    let config = glm52_test_config();
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(serde_json::from_str::<ModelConfig>(&json).unwrap(), config);
}

/// Test de débordement arithmétique : produits de dimensions.
#[test]
fn arithmetic_overflow_protection() {
    // Shape avec produit qui dépasse u64::MAX.
    let shape = Shape::new(vec![1 << 40, 1 << 24]).unwrap();
    assert!(shape.num_elements().is_err());

    // TensorMetadata avec overflow.
    let err = TensorMetadata::new("huge", shape, DType::F64).unwrap_err();
    assert!(matches!(err, CoreError::Overflow(_)));
}

/// Test des messages d'erreur français et explicites.
#[test]
fn error_messages_are_french_and_explicit() {
    // Dimension nulle.
    let err = Shape::new(vec![0]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("dimension"));
    assert!(msg.contains("0"));

    // Nom de tenseur vide.
    let err = TensorMetadata::new("", Shape::scalar(), DType::F32).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("nom"));

    // num_layers = 0.
    let mut config = glm52_test_config();
    config.num_layers = 0;
    let err = config.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("num_layers"));
}
