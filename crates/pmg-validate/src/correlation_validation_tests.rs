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

//! Tests unitaires pour la validation de corrélation.
//!
//! Ce module contient des tests spécifiques pour les fonctions
//! de corrélation, notamment `pearson_correlation`.

use super::correlation_validation::{
    covariance, pearson_correlation, standard_deviation, validate_correlation,
};

/// Test de corrélation de Pearson parfaite (positive).
///
/// Vérifie que des données parfaitement corrélées positivement
/// retournent un coefficient de 1.0.
#[test]
fn test_pearson_correlation_perfect() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let corr = pearson_correlation(&x, &y);
    assert!(corr.is_some());
    assert!((corr.unwrap() - 1.0).abs() < 1e-10);
}

/// Test de corrélation de Pearson parfaite (négative).
///
/// Vérifie que des données parfaitement corrélées négativement
/// retournent un coefficient de -1.0.
#[test]
fn test_pearson_correlation_negative() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let corr = pearson_correlation(&x, &y);
    assert!(corr.is_some());
    assert!((corr.unwrap() + 1.0).abs() < 1e-10);
}

/// Test de corrélation de Pearson avec des données non corrélées.
///
/// Vérifie que des données aléatoires sans corrélation retournent
/// un coefficient proche de 0.
#[test]
fn test_pearson_correlation_zero() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 1.0, 3.0, 2.0, 4.0];
    let corr = pearson_correlation(&x, &y);
    assert!(corr.is_some());
    // La corrélation n'est pas exactement 0 mais faible
    assert!(corr.unwrap().abs() < 0.5);
}

/// Test de corrélation avec des données constantes.
///
/// Vérifie que des données constantes retournent None (division par zéro).
#[test]
fn test_pearson_correlation_constant_data() {
    let x = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let corr = pearson_correlation(&x, &y);
    // L'écart-type de x est 0, donc la corrélation est indéfinie
    assert!(corr.is_none());
}

/// Test de covariance avec des données vides.
#[test]
fn test_covariance_empty() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];
    let cov = covariance(&x, &y);
    assert!(cov.is_none());
}

/// Test de covariance avec des données de tailles différentes.
#[test]
fn test_covariance_different_lengths() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    let cov = covariance(&x, &y);
    assert!(cov.is_none());
}

/// Test de validation de corrélation avec des données valides.
#[test]
fn test_validate_correlation_valid() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data_b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let result = validate_correlation("tensor_a", "tensor_b", &data_a, &data_b, 1.0, 0.01);

    assert_eq!(result.path_a, "tensor_a");
    assert_eq!(result.path_b, "tensor_b");
    assert!(result.observed_correlation > 0.99);
    assert!(result.issues.is_empty());
}

/// Test de validation de corrélation avec déviation significative.
#[test]
fn test_validate_correlation_deviation() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data_b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let result = validate_correlation("tensor_a", "tensor_b", &data_a, &data_b, 0.5, 0.01);

    assert!(!result.issues.is_empty());
    assert!(result.absolute_diff > 0.4);
}

/// Test de validation de corrélation avec des données vides.
#[test]
fn test_validate_correlation_empty_data() {
    let data_a: Vec<f64> = vec![];
    let data_b: Vec<f64> = vec![];
    let result = validate_correlation("tensor_a", "tensor_b", &data_a, &data_b, 0.5, 0.01);

    assert!(!result.issues.is_empty());
    assert_eq!(result.observed_correlation, 0.0);
}

/// Test de la fonction standard_deviation.
#[test]
fn test_standard_deviation_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let std = standard_deviation(&data);
    assert!(std.is_some());
    // Écart-type de [1,2,3,4,5] = sqrt(2) ≈ 1.4142
    assert!((std.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
}

/// Test de la fonction standard_deviation avec des données vides.
#[test]
fn test_standard_deviation_empty() {
    let data: Vec<f64> = vec![];
    let std = standard_deviation(&data);
    assert!(std.is_none());
}
