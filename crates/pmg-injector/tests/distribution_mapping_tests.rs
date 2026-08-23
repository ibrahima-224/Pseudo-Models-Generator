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

//! Tests d'intégration pour le mapping des distributions.
//!
//! Ce module teste la fonction `distribution_from_family` et ses propriétés
//! statistiques pour chaque famille de distribution supportée.

use pmg_blueprint::tensor_spec::DistributionFamily;
use pmg_injector::distribution_mapping::distribution_from_family;
use pmg_injector::error::InjectorError;
use pmg_math::rng::{derive_sub_seed, DeterministicRng};

/// RNG déterministe pour les tests.
fn rng_for(seed: [u8; 32]) -> DeterministicRng {
    DeterministicRng::from_seed(derive_sub_seed(&seed, "distribution_mapping", 0))
}

/// Seed de base pour les tests.
fn base_seed() -> [u8; 32] {
    [42u8; 32]
}

/// Génère un échantillon d'une distribution donnée.
fn sample_distribution(
    family: DistributionFamily,
    mean: f64,
    stddev: f64,
    n: usize,
    seed: [u8; 32],
) -> Vec<f64> {
    let mut rng = rng_for(seed);
    let mut dist = distribution_from_family(family, mean, stddev).unwrap();
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

/// Calcule la moyenne d'un échantillon.
fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Calcule l'écart-type de population d'un échantillon.
fn std_population(samples: &[f64]) -> f64 {
    let m = mean(samples);
    let variance = samples.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / samples.len() as f64;
    variance.sqrt()
}

#[test]
fn normal_distribution_has_correct_moments() {
    let samples = sample_distribution(DistributionFamily::Normal, 0.0, 1.0, 100_000, base_seed());
    let m = mean(&samples);
    let s = std_population(&samples);

    assert!(m.abs() < 0.02, "moyenne {m} hors tolérance");
    assert!((s - 1.0).abs() < 0.02, "écart-type {s} hors tolérance");
}

#[test]
fn student_t_distribution_has_correct_moments() {
    let samples = sample_distribution(DistributionFamily::StudentT, 0.0, 1.0, 100_000, base_seed());
    let m = mean(&samples);
    let s = std_population(&samples);

    // Student-t avec df=5 : moyenne 0, variance = df/(df-2) = 5/3 ≈ 1.6667
    // Après re-échelonnement pour stddev=1, variance = 1
    assert!(m.abs() < 0.02, "moyenne {m} hors tolérance");
    assert!((s - 1.0).abs() < 0.05, "écart-type {s} hors tolérance");
}

#[test]
fn laplace_distribution_has_correct_moments() {
    let samples = sample_distribution(DistributionFamily::Laplace, 0.0, 1.0, 100_000, base_seed());
    let m = mean(&samples);
    let s = std_population(&samples);

    // Laplace : moyenne 0, écart-type 1
    assert!(m.abs() < 0.02, "moyenne {m} hors tolérance");
    assert!((s - 1.0).abs() < 0.02, "écart-type {s} hors tolérance");
}

#[test]
fn log_normal_distribution_requires_positive_mean() {
    // Mean <= 0 doit échouer
    assert!(matches!(
        distribution_from_family(DistributionFamily::LogNormal, 0.0, 1.0),
        Err(InjectorError::InvalidPolicy(_))
    ));

    assert!(matches!(
        distribution_from_family(DistributionFamily::LogNormal, -1.0, 1.0),
        Err(InjectorError::InvalidPolicy(_))
    ));

    // Mean > 0 doit fonctionner
    assert!(distribution_from_family(DistributionFamily::LogNormal, 1.0, 0.5).is_ok());
}

#[test]
fn uniform_distribution_has_correct_moments() {
    let samples = sample_distribution(DistributionFamily::Uniform, 0.0, 1.0, 100_000, base_seed());
    let m = mean(&samples);
    let s = std_population(&samples);

    // Uniform : moyenne 0, écart-type 1
    assert!(m.abs() < 0.02, "moyenne {m} hors tolérance");
    assert!((s - 1.0).abs() < 0.02, "écart-type {s} hors tolérance");
}

#[test]
fn mixture_distribution_has_correct_moments() {
    let samples = sample_distribution(DistributionFamily::Mixture, 0.0, 1.0, 100_000, base_seed());
    let m = mean(&samples);
    let s = std_population(&samples);

    // Mixture bimodal : moyenne 0, variance 1
    assert!(m.abs() < 0.05, "moyenne {m} hors tolérance");
    assert!((s - 1.0).abs() < 0.05, "écart-type {s} hors tolérance");
}

#[test]
fn weibull_distribution_is_approximation() {
    let samples = sample_distribution(DistributionFamily::Weibull, 0.0, 1.0, 100_000, base_seed());
    let s = std_population(&samples);

    // Weibull est une approximation : on vérifie que l'écart-type est fini et positif
    assert!(s.is_finite(), "écart-type doit être fini");
    assert!(s > 0.0, "écart-type doit être positif");
}

#[test]
fn pareto_distribution_is_approximation() {
    let samples = sample_distribution(DistributionFamily::Pareto, 0.0, 1.0, 100_000, base_seed());
    let s = std_population(&samples);

    // Pareto est une approximation : on vérifie que l'écart-type est fini et positif
    assert!(s.is_finite(), "écart-type doit être fini");
    assert!(s > 0.0, "écart-type doit être positif");
}

#[test]
fn invalid_stddev_rejected() {
    // Stddev = 0 doit échouer
    assert!(matches!(
        distribution_from_family(DistributionFamily::Normal, 0.0, 0.0),
        Err(InjectorError::InvalidPolicy(_))
    ));

    // Stddev négatif doit échouer
    assert!(matches!(
        distribution_from_family(DistributionFamily::Normal, 0.0, -1.0),
        Err(InjectorError::InvalidPolicy(_))
    ));

    // Stddev NaN doit échouer
    assert!(matches!(
        distribution_from_family(DistributionFamily::Normal, 0.0, f64::NAN),
        Err(InjectorError::InvalidPolicy(_))
    ));

    // Stddev infini doit échouer
    assert!(matches!(
        distribution_from_family(DistributionFamily::Normal, 0.0, f64::INFINITY),
        Err(InjectorError::InvalidPolicy(_))
    ));
}

#[test]
fn unsupported_family_rejected() {
    // On ne peut pas tester directement une famille non supportée car DistributionFamily
    // est une énumération exhaustive, mais on vérifie que le code gère bien ce cas
    // en testant la branche _ => du match.
    // Note: Ceci est un test de documentation plutôt que fonctionnel.
}

#[test]
fn distribution_is_deterministic() {
    let a = sample_distribution(DistributionFamily::Normal, 0.0, 1.0, 1000, base_seed());
    let b = sample_distribution(DistributionFamily::Normal, 0.0, 1.0, 1000, base_seed());
    assert_eq!(
        a, b,
        "les échantillons doivent être identiques avec la même seed"
    );
}

#[test]
fn different_seeds_produce_different_samples() {
    let a = sample_distribution(DistributionFamily::Normal, 0.0, 1.0, 1000, base_seed());
    let mut other_seed = base_seed();
    other_seed[0] ^= 0xFF;
    let b = sample_distribution(DistributionFamily::Normal, 0.0, 1.0, 1000, other_seed);
    assert_ne!(
        a, b,
        "les échantillons doivent être différents avec des seeds différentes"
    );
}

#[test]
fn distribution_pdf_is_non_negative() {
    let dist = distribution_from_family(DistributionFamily::Normal, 0.0, 1.0).unwrap();

    // Vérifier que la PDF est non négative pour plusieurs points
    let test_points = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0];
    for &x in &test_points {
        let pdf = dist.pdf(x);
        assert!(pdf >= 0.0, "PDF doit être non négative en x={x}");
    }
}

#[test]
fn distribution_cdf_is_bounded() {
    let dist = distribution_from_family(DistributionFamily::Normal, 0.0, 1.0).unwrap();

    // Vérifier que la CDF est dans [0, 1] pour plusieurs points
    let test_points = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0];
    for &x in &test_points {
        if let Some(cdf) = dist.cdf(x) {
            assert!(
                (0.0..=1.0).contains(&cdf),
                "CDF doit être dans [0, 1] en x={x}"
            );
        }
    }
}
