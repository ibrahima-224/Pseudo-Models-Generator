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

//! Tests unitaires pour le validateur principal.

use super::*;
use crate::severity::Severity;
use crate::types::*;

#[test]
fn validation_summary_is_valid() {
    let mut summary = ValidationSummary::default();
    assert!(summary.is_valid());

    summary.info_count = 5;
    assert!(summary.is_valid());

    summary.warning_count = 2;
    assert!(summary.is_valid());

    summary.error_count = 1;
    assert!(!summary.is_valid());
}

#[test]
fn validation_summary_total_issues() {
    let summary = ValidationSummary {
        info_count: 1,
        warning_count: 2,
        error_count: 3,
        critical_count: 4,
    };
    assert_eq!(summary.total_issues(), 10);
}

#[test]
fn validate_tensor_empty_data() {
    let validator = ModelValidator::default();
    let result = validator.validate_tensor("test", &[], None, None);
    assert!(!result.issues.is_empty());
    assert!(result.issues.iter().any(|i| i.severity == Severity::Error));
}

#[test]
fn validate_tensor_normal_data() {
    let validator = ModelValidator::default();
    let data = [0.0, 1.0, 2.0, 3.0, 4.0];
    let result = validator.validate_tensor("test", &data, None, None);
    // Pas d'erreurs critiques
    assert!(!result
        .issues
        .iter()
        .any(|i| i.severity == Severity::Critical));
}

#[test]
fn validate_tensor_with_expected_mean() {
    let validator = ModelValidator::default();
    let data = [0.0, 1.0, 2.0, 3.0, 4.0]; // mean = 2.0
    let result = validator.validate_tensor("test", &data, Some(2.0), None);
    // Pas de warning sur la moyenne
    assert!(!result.issues.iter().any(|i| {
        i.category == ValidationCategory::Statistical && i.severity == Severity::Warning
    }));
}

#[test]
fn validate_model_multiple_tensors() {
    let validator = ModelValidator::default();
    let tensors = [
        ("layer1", &[0.0, 1.0, 2.0] as &[f64], None, None),
        ("layer2", &[3.0, 4.0, 5.0] as &[f64], None, None),
    ];
    let result = validator.validate_model("test_model", &tensors);
    assert_eq!(result.tensor_count, 2);
    assert!(result.summary.is_valid());
}
