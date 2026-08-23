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

//! Tests d'intégration pour le crate pmg-validate.
//!
//! Ce module contient des tests d'intégration qui valident le comportement
//! du crate pmg-validate dans son ensemble.

use pmg_validate::{
    validate_correlation, validate_distribution, validate_dtype, validate_outliers, validate_shape,
    validate_statistics, Distribution, ModelValidator, Severity, SimpleDType, ValidationConfig,
};

#[test]
fn test_complete_validation_workflow() {
    // Configuration de validation
    let config = ValidationConfig::default();
    let _validator = ModelValidator::new(config);

    // Données de test pour un tenseur
    let tensor_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();

    // Validation structurelle
    let shape_result = validate_shape("layer1.weight", &[10, 10], &[10, 10]);
    assert!(shape_result.issues.is_empty());

    // Validation de type
    let dtype_result = validate_dtype("layer1.weight", SimpleDType::F32, SimpleDType::F32);
    assert!(dtype_result.issues.is_empty());

    // Validation statistique
    let stat_result = validate_statistics(
        "layer1.weight",
        0.5,    // moyenne observée
        0.5,    // moyenne cible
        0.2887, // écart-type observé (approximatif)
        0.2887, // écart-type cible
        0.1,    // tolérance
    );
    assert!(stat_result.issues.is_empty());

    // Validation de distribution
    let dist_result =
        validate_distribution("layer1.weight", &tensor_data, Distribution::Normal, 0.05);
    // Peut avoir des avertissements selon les données
    assert_eq!(dist_result.distribution, Distribution::Normal);

    // Validation des outliers
    let outlier_result = validate_outliers("layer1.weight", &tensor_data, 3.0, 0.1);
    assert!(outlier_result.outlier_count == 0);

    // Validation de corrélation (avec deux tenseurs identiques)
    let corr_result = validate_correlation(
        "layer1.weight",
        "layer1.bias",
        &tensor_data,
        &tensor_data,
        1.0, // corrélation attendue
        0.1, // tolérance
    );
    assert!(corr_result.observed_correlation > 0.99);
}

#[test]
fn test_validator_with_multiple_tensors() {
    let validator = ModelValidator::default();

    // Simuler plusieurs tenseurs
    let tensors = vec![
        ("layer1.weight", vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        ("layer1.bias", vec![0.01, 0.02, 0.03, 0.04, 0.05]),
        ("layer2.weight", vec![0.5, 0.4, 0.3, 0.2, 0.1]),
    ];

    let mut all_results = Vec::new();

    for (path, data) in tensors {
        let result = validator.validate_tensor(path, &data, Some(0.3), Some(0.1));
        all_results.push(result);
    }

    // Vérifier que tous les tenseurs ont été validés
    assert_eq!(all_results.len(), 3);

    // Vérifier qu'il n'y a pas d'erreurs critiques
    for result in &all_results {
        for issue in &result.issues {
            assert_ne!(issue.severity, Severity::Critical);
        }
    }
}

#[test]
fn test_validation_with_errors() {
    // Test avec des données qui génèrent des erreurs
    let empty_data: Vec<f64> = vec![];
    let result = validate_distribution("empty_tensor", &empty_data, Distribution::Normal, 0.05);
    assert!(!result.issues.is_empty());
    assert_eq!(result.issues[0].severity, Severity::Error);
}

#[test]
fn test_validation_configurations() {
    // Test avec différentes configurations
    let configs = vec![
        ValidationConfig::default(),
        ValidationConfig {
            outlier_threshold: 2.0,
            energy_threshold: 0.8,
            statistical_tolerance: 0.05,
            check_structural: true,
            check_statistical: true,
            check_distribution: true,
            check_outliers: true,
            check_metadata: true,
        },
    ];

    for config in configs {
        let validator = ModelValidator::new(config);
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = validator.validate_tensor("test_tensor", &data, Some(0.3), Some(0.1));
        // La validation doit fonctionner quelle que soit la configuration
        assert!(result.path == "test_tensor");
    }
}
