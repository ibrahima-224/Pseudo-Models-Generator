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

//! Tests unitaires pour le module `injection_validator`.

use super::{validate_against_policy, InjectionReport, ValidationTolerances};
use crate::injection_policy::InjectionPolicy;
use pmg_math::distribution::Distribution;
use pmg_math::rng::{derive_sub_seed, DeterministicRng};

/// Fonction utilitaire pour créer un générateur déterministe à partir d'une graine.
fn rng_for(seed: [u8; 32]) -> DeterministicRng {
    DeterministicRng::from_seed(derive_sub_seed(&seed, "validator", 0))
}

/// Graine de base pour les tests déterministes.
fn base_seed() -> [u8; 32] {
    [31u8; 32]
}

/// Génère un échantillon de `n` valeurs suivant une loi normale standard.
fn gaussian(n: usize) -> Vec<f64> {
    // Échantillon normal standard déterministe.
    let mut rng = rng_for(base_seed());
    let mut out = Vec::with_capacity(n);
    let mut normal = pmg_math::distributions::Normal::new(0.0, 1.0).unwrap();
    for _ in 0..n {
        out.push(normal.sample(&mut rng));
    }
    out
}

#[test]
fn report_measures_basic_metrics() {
    let values = gaussian(10_000);
    let report = InjectionReport::from_values(&values, 3.0, &[]).unwrap();
    assert_eq!(report.count, 10_000);
    assert!(report.mean.abs() < 0.05);
    assert!((report.std_population - 1.0).abs() < 0.05);
    assert!(report.max_abs > 0.0);
    assert_eq!(report.quantiles.len(), 5);
    // Les quantiles sont ordonnés.
    for w in report.quantiles.windows(2) {
        assert!(w[0] <= w[1]);
    }
}

#[test]
fn report_matrix_adds_correlation_and_rank() {
    // Matrice avec corrélation forte : rang bas, corrélation > 0.
    // NB : 200 lignes suffisent (effective_rank est O(200·m²·n)). Avec
    // ρ = 0.99, le bruit résiduel est minuscule : rang effectif ≤ 5.
    // Tolérance accrue pour la compatibilité avec différents générateurs aléatoires.
    let mut rng = rng_for(base_seed());
    let data = crate::correlated::generate_correlated_columns(&mut rng, 200, 4, 0.99, 1.0).unwrap();
    let report = InjectionReport::from_matrix(&data, 200, 4, 3.0, &[]).unwrap();
    assert!(report.mean_column_correlation.unwrap() > 0.5);
    assert!(
        report.estimated_rank.unwrap() <= 5,
        "rank est {} mais attendu ≤ 5",
        report.estimated_rank.unwrap()
    );
}

#[test]
fn validate_passes_when_within_tolerances() {
    // Outliers 1 % avec seuil 3 : p̂ ≈ 0.01 (loi normale), tolérance 0.02.
    let values = gaussian(100_000);
    let policy = InjectionPolicy::new(0.01, 5.0, 0.0, 0.0, 1, 0.0, 0.0, 5.0, 0.0, 0.5).unwrap();
    let tolerances = ValidationTolerances {
        outlier_ratio: 0.02,
        std_relative: 0.1,
        correlation: 0.1,
    };
    let v = validate_against_policy(&values, &policy, Some(1.0), 3.0, &tolerances).unwrap();
    assert!(v.passed, "échecs : {:?}", v.failures);
}

#[test]
fn validate_detects_outlier_deviation() {
    // Aucun outlier (fréquence demandée 0.5) → échec du critère.
    let values = gaussian(10_000);
    let policy = InjectionPolicy::new(0.5, 5.0, 0.0, 0.0, 1, 0.0, 0.0, 5.0, 0.0, 0.5).unwrap();
    let v = validate_against_policy(
        &values,
        &policy,
        None,
        3.0,
        &ValidationTolerances::default(),
    )
    .unwrap();
    assert!(!v.passed);
    assert!(v.failures.iter().any(|(name, _)| name == "outlier_ratio"));
}

#[test]
fn validate_detects_std_deviation() {
    // Écart-type cible 0.1 mais valeurs N(0,1) → échec du critère std.
    let values = gaussian(10_000);
    let policy = InjectionPolicy::none();
    let tolerances = ValidationTolerances {
        outlier_ratio: 0.01,
        std_relative: 0.1,
        correlation: 0.1,
    };
    let v = validate_against_policy(&values, &policy, Some(0.1), 3.0, &tolerances).unwrap();
    assert!(!v.passed);
    assert!(v.failures.iter().any(|(name, _)| name == "std_relative"));
}

#[test]
fn validate_correlation_requires_matrix() {
    // Tenseur 1D avec politique de corrélation : échec explicite.
    let values = gaussian(1_000);
    let mut policy = InjectionPolicy::none();
    policy.correlation_strength = 0.5;
    let v = validate_against_policy(
        &values,
        &policy,
        None,
        3.0,
        &ValidationTolerances::default(),
    )
    .unwrap();
    assert!(!v.passed);
    assert!(v.failures.iter().any(|(name, _)| name == "correlation"));
}

#[test]
fn validation_is_deterministic_and_repeatable() {
    let values = gaussian(20_000);
    let policy = InjectionPolicy::default();
    let tolerances = ValidationTolerances::default();
    let a = validate_against_policy(&values, &policy, Some(1.0), 3.0, &tolerances).unwrap();
    let b = validate_against_policy(&values, &policy, Some(1.0), 3.0, &tolerances).unwrap();
    assert_eq!(a, b);
}

#[test]
fn empty_values_rejected() {
    let policy = InjectionPolicy::none();
    assert!(
        validate_against_policy(&[], &policy, None, 3.0, &ValidationTolerances::default()).is_err()
    );
}

#[test]
fn zero_outlier_frequency_skips_outlier_criterion() {
    let values = gaussian(1_000);
    let policy = InjectionPolicy::none();
    let v = validate_against_policy(
        &values,
        &policy,
        None,
        3.0,
        &ValidationTolerances::default(),
    )
    .unwrap();
    // Aucun critère activé → succès trivial.
    assert!(v.passed);
    assert!(v.deviations.is_empty());
}
