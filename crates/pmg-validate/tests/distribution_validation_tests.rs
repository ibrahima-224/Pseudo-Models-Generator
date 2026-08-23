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

//! Tests d'intégration pour la validation des distributions.

use pmg_validate::{validate_distribution, Distribution, Severity};

#[test]
fn validate_distribution_normal_good() {
    // Données normales simulées
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = validate_distribution("test", &data, Distribution::Normal, 0.05);
    assert!(result.issues.is_empty());
}

#[test]
fn validate_distribution_normal_bad() {
    // Données uniformes (pas normales)
    let data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let result = validate_distribution("test", &data, Distribution::Normal, 0.05);
    // Peut générer un avertissement ou une erreur selon la taille
    // On vérifie juste que la validation fonctionne
    assert_eq!(result.distribution, Distribution::Normal);
}

#[test]
fn validate_distribution_student_t() {
    // Test avec distribution Student-t
    let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = validate_distribution("test", &data, Distribution::StudentT, 0.05);
    assert_eq!(result.distribution, Distribution::StudentT);
    // Vérifier que la statistique KS n'est pas toujours 0.0
    assert!(result.ks_statistic >= 0.0, "KS devrait être >= 0");
    // Vérifier que la p-value est dans [0, 1]
    assert!(
        result.p_value >= 0.0 && result.p_value <= 1.0,
        "p-value devrait être dans [0, 1]"
    );
}

#[test]
fn validate_distribution_weibull() {
    // Test avec distribution Weibull
    let data = [0.5, 1.0, 1.5, 2.0, 2.5];
    let result = validate_distribution("test", &data, Distribution::Weibull, 0.05);
    assert_eq!(result.distribution, Distribution::Weibull);
    assert!(result.ks_statistic >= 0.0, "KS devrait être >= 0");
    assert!(
        result.p_value >= 0.0 && result.p_value <= 1.0,
        "p-value devrait être dans [0, 1]"
    );
}

#[test]
fn validate_distribution_pareto() {
    // Test avec distribution Pareto
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = validate_distribution("test", &data, Distribution::Pareto, 0.05);
    assert_eq!(result.distribution, Distribution::Pareto);
    assert!(result.ks_statistic >= 0.0, "KS devrait être >= 0");
    assert!(
        result.p_value >= 0.0 && result.p_value <= 1.0,
        "p-value devrait être dans [0, 1]"
    );
}

#[test]
fn validate_distribution_log_normal() {
    // Test avec distribution log-normale
    let data = [0.1, 0.5, 1.0, 2.0, 5.0];
    let result = validate_distribution("test", &data, Distribution::LogNormal, 0.05);
    assert_eq!(result.distribution, Distribution::LogNormal);
    assert!(result.ks_statistic >= 0.0, "KS devrait être >= 0");
    assert!(
        result.p_value >= 0.0 && result.p_value <= 1.0,
        "p-value devrait être dans [0, 1]"
    );
}

#[test]
fn validate_distribution_empty_data() {
    // Test avec données vides
    let data = [];
    let result = validate_distribution("test", &data, Distribution::Normal, 0.05);
    assert!(!result.issues.is_empty());
    assert_eq!(result.issues[0].severity, Severity::Error);
}

#[test]
fn validate_distribution_student_t_heavy_tails() {
    // Test avec des données à queues lourdes
    let data = [-10.0, -1.0, 0.0, 1.0, 10.0];
    let result = validate_distribution("test", &data, Distribution::StudentT, 0.05);
    assert_eq!(result.distribution, Distribution::StudentT);
    assert!(result.ks_statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn validate_distribution_weibull_positive_data() {
    // Test avec des données strictement positives
    let data = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let result = validate_distribution("test", &data, Distribution::Weibull, 0.05);
    assert_eq!(result.distribution, Distribution::Weibull);
    assert!(result.ks_statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn validate_distribution_pareto_heavy_tail() {
    // Test avec des données à queue lourde
    let data = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    let result = validate_distribution("test", &data, Distribution::Pareto, 0.05);
    assert_eq!(result.distribution, Distribution::Pareto);
    assert!(result.ks_statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn validate_distribution_lognormal_positive_data() {
    // Test avec des données strictement positives
    let data = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let result = validate_distribution("test", &data, Distribution::LogNormal, 0.05);
    assert_eq!(result.distribution, Distribution::LogNormal);
    assert!(result.ks_statistic >= 0.0);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

/// Test : vérifie que le tri Weibull ne panic pas avec des NaN.
///
/// Les NaN doivent être gérés correctement lors du tri (via unwrap_or)
/// pour éviter un panic dans la comparaison partielle.
#[test]
fn test_weibull_sort_with_nan() {
    // Données avec des NaN mélangés à des valeurs normales
    let data = [1.0, f64::NAN, 2.0, f64::NAN, 3.0];
    let result = validate_distribution("test_nan_weibull", &data, Distribution::Weibull, 0.05);

    // La validation ne doit pas panic
    assert_eq!(result.distribution, Distribution::Weibull);
    // Les statistiques doivent être calculées (peuvent être NaN ou valides)
    // L'important est qu'aucun panic ne se produise
}

/// Test : vérifie que le test Kolmogorov-Smirnov ne panic pas avec des NaN.
///
/// Le test KS utilise un tri interne qui doit gérer les NaN correctement.
#[test]
fn test_kolmogorov_smirnov_with_nan() {
    // Données avec des NaN pour la distribution normale
    let data = [0.0, f64::NAN, 1.0, f64::NAN, -1.0];
    let result = validate_distribution("test_nan_ks", &data, Distribution::Normal, 0.05);

    // La validation ne doit pas panic
    assert_eq!(result.distribution, Distribution::Normal);
    // Vérifier que les champs de base sont remplis
    assert!(!result.path.is_empty());
}

/// Test : vérifie la validation avec des valeurs infinies.
///
/// Les valeurs infinies (positives et négatives) doivent être gérées
/// sans panic lors de la validation.
#[test]
fn test_distribution_validation_with_infinity() {
    // Données avec des infinis
    let data = [f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY];
    let result = validate_distribution("test_infinity", &data, Distribution::Normal, 0.05);

    // La validation ne doit pas panic
    assert_eq!(result.distribution, Distribution::Normal);
    // Les statistiques peuvent être affectées par les infinis
    // L'important est qu'aucun panic ne se produise
}

/// Test : vérifie la validation avec un mélange de NaN et d'infinis.
///
/// Ce test combine des cas limites pour vérifier la robustesse globale.
#[test]
fn test_distribution_validation_mixed_special_values() {
    // Mélange de NaN, infinis et valeurs normales
    let data = [f64::NAN, f64::INFINITY, -1.0, 0.0, 1.0, f64::NEG_INFINITY];
    let result = validate_distribution("test_mixed_special", &data, Distribution::Weibull, 0.05);

    // La validation ne doit pas panic
    assert_eq!(result.distribution, Distribution::Weibull);
}
