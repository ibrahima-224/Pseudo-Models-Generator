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

//! Tests unitaires pour les helpers statistiques.
//!
//! Ce module contient des tests pour les fonctions utilitaires
//! de calcul statistique, notamment `compute_basic_stats`.

use super::statistical_helpers::{
    calculate_mean, calculate_std, compute_basic_stats, relative_error,
};

/// Test de calcul des statistiques de base avec des données valides.
///
/// Vérifie que la fonction retourne correctement la moyenne,
/// la variance et l'écart-type pour un ensemble de données simple.
#[test]
fn test_compute_basic_stats_valid_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (mean, variance, std) = compute_basic_stats(&data);

    // Moyenne attendue : (1+2+3+4+5)/5 = 3.0
    assert!((mean - 3.0).abs() < 1e-10);

    // Variance attendue : ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²)/5 = 2.0
    assert!((variance - 2.0).abs() < 1e-10);

    // Écart-type attendu : sqrt(2) ≈ 1.4142
    assert!((std - std::f64::consts::SQRT_2).abs() < 1e-6);
}

/// Test de calcul des statistiques de base avec des données vides.
///
/// Vérifie que la fonction retourne (0.0, 0.0, 0.0) pour des données vides.
#[test]
fn test_compute_basic_stats_empty_data() {
    let data: Vec<f64> = vec![];
    let (mean, variance, std) = compute_basic_stats(&data);

    assert_eq!(mean, 0.0);
    assert_eq!(variance, 0.0);
    assert_eq!(std, 0.0);
}

/// Test de calcul des statistiques de base avec une seule valeur.
///
/// Vérifie que la variance est 0 pour une seule valeur.
#[test]
fn test_compute_basic_stats_single_value() {
    let data = vec![5.0];
    let (mean, variance, std) = compute_basic_stats(&data);

    assert_eq!(mean, 5.0);
    assert_eq!(variance, 0.0);
    assert_eq!(std, 0.0);
}

/// Test de calcul des statistiques de base avec des valeurs négatives.
///
/// Vérifie le calcul correct avec des nombres négatifs.
#[test]
fn test_compute_basic_stats_negative_values() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let (mean, variance, std) = compute_basic_stats(&data);

    // Moyenne attendue : 0.0
    assert!((mean - 0.0).abs() < 1e-10);

    // Variance attendue : ((-2)² + (-1)² + 0² + 1² + 2²)/5 = 2.0
    assert!((variance - 2.0).abs() < 1e-10);

    // Écart-type attendu : sqrt(2) ≈ 1.4142
    assert!((std - std::f64::consts::SQRT_2).abs() < 1e-6);
}

/// Test de calcul des statistiques de base avec des valeurs identiques.
///
/// Vérifie que la variance est 0 pour des valeurs identiques.
#[test]
fn test_compute_basic_stats_identical_values() {
    let data = vec![3.0, 3.0, 3.0, 3.0, 3.0];
    let (mean, variance, std) = compute_basic_stats(&data);

    assert_eq!(mean, 3.0);
    assert_eq!(variance, 0.0);
    assert_eq!(std, 0.0);
}

/// Test de la fonction relative_error.
///
/// Vérifie le calcul de l'erreur relative entre deux valeurs.
#[test]
fn test_relative_error_basic() {
    let observed = 10.5;
    let target = 10.0;
    let error = relative_error(observed, target);

    // |10.5 - 10.0| / max(|10.0|, 1e-10) = 0.5 / 10.0 = 0.05
    assert!((error - 0.05).abs() < 1e-10);
}

/// Test de la fonction relative_error avec une valeur cible nulle.
///
/// Vérifie que la fonction utilise l'epsilon pour éviter la division par zéro.
#[test]
fn test_relative_error_zero_target() {
    let observed = 5.0;
    let target = 0.0;
    let error = relative_error(observed, target);

    // |5.0 - 0.0| / max(|0.0|, 1e-10) = 5.0 / 1e-10 = 5e10
    assert!((error - 5e10).abs() < 1e5);
}

/// Test de la fonction calculate_mean.
///
/// Vérifie le calcul de la moyenne.
#[test]
fn test_calculate_mean_valid() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = calculate_mean(&data);

    assert!(mean.is_some());
    assert!((mean.unwrap() - 3.0).abs() < 1e-10);
}

/// Test de la fonction calculate_mean avec des données vides.
///
/// Vérifie que la fonction retourne None pour des données vides.
#[test]
fn test_calculate_mean_empty() {
    let data: Vec<f64> = vec![];
    let mean = calculate_mean(&data);

    assert!(mean.is_none());
}

/// Test de la fonction calculate_std.
///
/// Vérifie le calcul de l'écart-type.
#[test]
fn test_calculate_std_valid() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let std = calculate_std(&data);

    assert!(std.is_some());
    assert!((std.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
}

/// Test de la fonction calculate_std avec une seule valeur.
///
/// Vérifie que la fonction retourne None pour une seule valeur.
#[test]
fn test_calculate_std_single_value() {
    let data = vec![5.0];
    let std = calculate_std(&data);

    assert!(std.is_none());
}
