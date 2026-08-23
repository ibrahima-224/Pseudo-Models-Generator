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

//! Tests de validation statistique des distributions de probabilité.
//!
//! Ces tests vérifient les propriétés statistiques essentielles de chaque
//! distribution implémentée dans `pmg_math::distributions`. Les tests
//! utilisent des échantillons de taille suffisante pour des estimations
//! fiables et des tolérances statistiques appropriées.
//!
//! Conformité : `docs/architecture/09-tests-benchmarks-ci.md` §1.7.

use pmg_core::distribution_config::DistributionConfig;
use pmg_math::distribution::from_config;
use pmg_math::rng::DeterministicRng;
use pmg_math::statistics;

/// Taille d'échantillon pour les tests statistiques.
/// Doit être suffisamment grande pour des estimations fiables.
const SAMPLE_SIZE: usize = 10_000;

/// Tolerance relative pour les tests de variance (10%).
const VARIANCE_TOLERANCE: f64 = 0.10;

/// Tolerance relative pour les tests de skewness (20%).
const SKEWNESS_TOLERANCE: f64 = 0.20;

/// Tolerance relative pour les tests de kurtosis (30%).
const KURTOSIS_TOLERANCE: f64 = 0.30;

/// Génère des échantillons selon une configuration donnée.
fn generate_samples(config: &DistributionConfig, n: usize) -> Vec<f64> {
    let mut dist = from_config(config).expect("configuration valide");
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

/// Vérifie que la moyenne empirique est dans la tolérance attendue.
fn assert_mean(samples: &[f64], expected_mean: f64, expected_std: f64) {
    let n = samples.len() as f64;
    let mu_hat = statistics::mean(samples).expect("calcul de moyenne");
    let tolerance = 5.0 * expected_std / n.sqrt();
    let err = (mu_hat - expected_mean).abs();
    assert!(
        err <= tolerance,
        "moyenne hors tolérance : μ̂={mu_hat:.6} attendu {expected_mean:.6}, \
         tolérance {tolerance:.6}, écart {err:.6}"
    );
}

/// Vérifie que l'écart-type empirique est dans la tolérance attendue.
fn assert_variance(samples: &[f64], expected_std: f64) {
    let s_hat = statistics::std_population(samples).expect("calcul d'écart-type");
    let rel = (s_hat - expected_std).abs() / expected_std;
    assert!(
        rel <= VARIANCE_TOLERANCE,
        "écart-type hors tolérance : σ̂={s_hat:.6} attendu {expected_std:.6}, rel={rel:.6}"
    );
}

/// Calcule le skewness empirique d'un échantillon.
fn skewness(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = statistics::mean(samples).unwrap();
    let std = statistics::std_population(samples).unwrap();
    let m3 = samples.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
    m3 / std.powi(3)
}

/// Calcule le kurtosis empirique (excès) d'un échantillon.
fn kurtosis_excess(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = statistics::mean(samples).unwrap();
    let std = statistics::std_population(samples).unwrap();
    let m4 = samples.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    m4 / std.powi(4) - 3.0
}

/// Vérifie que le quantile empirique est proche du quantile théorique.
fn assert_quantile(samples: &[f64], theoretical_q: f64, empirical_q: f64, tolerance: f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (empirical_q * (sorted.len() - 1) as f64).round() as usize;
    let empirical_value = sorted[idx];
    let err = (empirical_value - theoretical_q).abs();
    assert!(
        err <= tolerance,
        "quantile {empirical_q} hors tolérance : théorique {theoretical_q:.6}, \
         empirique {empirical_value:.6}, écart {err:.6}"
    );
}

// ============================================================================
// Tests de la distribution Normale
// ============================================================================

#[test]
fn normal_mean_variance() {
    let config = DistributionConfig::normal(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    assert_mean(&samples, 0.0, 1.0);
    assert_variance(&samples, 1.0);
}

#[test]
fn normal_quantiles() {
    let config = DistributionConfig::normal(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Quantiles théoriques pour N(0,1)
    assert_quantile(&samples, -1.2816, 0.10, 0.05); // 10%
    assert_quantile(&samples, 0.0, 0.50, 0.05); // médiane
    assert_quantile(&samples, 1.2816, 0.90, 0.05); // 90%
}

#[test]
fn normal_skewness_kurtosis() {
    let config = DistributionConfig::normal(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    let skew = skewness(&samples);
    let kurt = kurtosis_excess(&samples);
    assert!(
        skew.abs() <= SKEWNESS_TOLERANCE,
        "skewness normale hors tolérance : {skew:.6}"
    );
    assert!(
        kurt.abs() <= KURTOSIS_TOLERANCE,
        "kurtosis excès normale hors tolérance : {kurt:.6}"
    );
}

// ============================================================================
// Tests de la distribution Student-t
// ============================================================================

#[test]
fn student_t_mean_variance() {
    let config = DistributionConfig::student_t(5.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Student-t(5) : moyenne = 0, variance = df/(df-2) = 5/3
    let expected_var: f64 = 5.0 / 3.0;
    assert_mean(&samples, 0.0, expected_var.sqrt());
    assert_variance(&samples, expected_var.sqrt());
}

#[test]
fn student_t_quantiles() {
    let config = DistributionConfig::student_t(5.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Quantiles approximatifs pour t(5)
    assert_quantile(&samples, -1.4759, 0.10, 0.10);
    assert_quantile(&samples, 0.0, 0.50, 0.10);
    assert_quantile(&samples, 1.4759, 0.90, 0.10);
}

#[test]
fn student_t_heavy_tails() {
    let config = DistributionConfig::student_t(3.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Student-t(3) a des queues plus lourdes que la normale
    let extreme_count = samples.iter().filter(|&&x| x.abs() > 3.0).count();
    let extreme_ratio = extreme_count as f64 / SAMPLE_SIZE as f64;
    // Pour t(3), P(|X| > 3) ≈ 0.04
    assert!(
        extreme_ratio > 0.02,
        "queues trop légères : {extreme_ratio:.4}"
    );
    assert!(
        extreme_ratio < 0.10,
        "queues trop lourdes : {extreme_ratio:.4}"
    );
}

// ============================================================================
// Tests de la distribution Laplace
// ============================================================================

#[test]
fn laplace_mean_variance() {
    let config = DistributionConfig::laplace(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Laplace(μ, b) : moyenne = μ, variance = 2b²
    let expected_var: f64 = 2.0;
    assert_mean(&samples, 0.0, expected_var.sqrt());
    assert_variance(&samples, expected_var.sqrt());
}

#[test]
fn laplace_skewness_kurtosis() {
    let config = DistributionConfig::laplace(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    let skew = skewness(&samples);
    let kurt = kurtosis_excess(&samples);
    // Laplace : skewness = 0, kurtosis excès = 3
    assert!(
        skew.abs() <= SKEWNESS_TOLERANCE,
        "skewness Laplace hors tolérance : {skew:.6}"
    );
    assert!(
        (kurt - 3.0).abs() <= 2.0 * KURTOSIS_TOLERANCE,
        "kurtosis excès Laplace hors tolérance : {kurt:.6}"
    );
}

// ============================================================================
// Tests de la distribution Log-Normale
// ============================================================================

#[test]
fn log_normal_mean_variance() {
    let config = DistributionConfig::log_normal(0.0, 0.5);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // LogN(μ, σ) : moyenne = exp(μ + σ²/2), variance = [exp(σ²)-1] * exp(2μ + σ²)
    let mu: f64 = 0.0;
    let sigma: f64 = 0.5;
    let expected_mean = (mu + sigma * sigma / 2.0).exp();
    let expected_var = ((sigma * sigma).exp() - 1.0) * (2.0 * mu + sigma * sigma).exp();
    assert_mean(&samples, expected_mean, expected_var.sqrt());
    assert_variance(&samples, expected_var.sqrt());
}

#[test]
fn log_normal_positive_values() {
    let config = DistributionConfig::log_normal(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Toutes les valeurs doivent être positives
    let all_positive = samples.iter().all(|&x| x > 0.0);
    assert!(
        all_positive,
        "log-normale doit générer uniquement des valeurs positives"
    );
}

// ============================================================================
// Tests de la distribution Weibull
// ============================================================================

#[test]
fn weibull_mean_variance() {
    let config = DistributionConfig::weibull(1.0, 2.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Weibull(λ, k) : moyenne = λ * Γ(1 + 1/k)
    // Pour λ=1, k=2 : Γ(1.5) ≈ 0.8862
    let expected_mean: f64 = 1.0 * 0.8862;
    assert_mean(&samples, expected_mean, 0.2); // tolérance augmentée
                                               // Variance = λ² * [Γ(1 + 2/k) - (Γ(1 + 1/k))²]
                                               // Pour λ=1, k=2 : variance = 1 * [1 - 0.8862²] ≈ 0.215
    let expected_var: f64 = 0.215;
    assert_variance(&samples, expected_var.sqrt());
}

#[test]
fn weibull_positive_values() {
    let config = DistributionConfig::weibull(2.0, 1.5);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    let all_positive = samples.iter().all(|&x| x >= 0.0);
    assert!(
        all_positive,
        "Weibull doit générer uniquement des valeurs non négatives"
    );
}

// ============================================================================
// Tests de la distribution Pareto
// ============================================================================

#[test]
fn pareto_mean_variance() {
    let config = DistributionConfig::pareto(1.0, 3.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Pareto(x_m, α) : moyenne = α * x_m / (α - 1) pour α > 1
    // variance = x_m² * α / [(α - 1)² * (α - 2)] pour α > 2
    let xm: f64 = 1.0;
    let alpha: f64 = 3.0;
    let expected_mean = alpha * xm / (alpha - 1.0);
    let expected_var = xm * xm * alpha / ((alpha - 1.0).powi(2) * (alpha - 2.0));
    assert_mean(&samples, expected_mean, expected_var.sqrt());
    assert_variance(&samples, expected_var.sqrt());
}

#[test]
fn pareto_heavy_tails() {
    let config = DistributionConfig::pareto(1.0, 2.5);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Pareto a des queues très lourdes
    let extreme_count = samples.iter().filter(|&&x| x > 5.0).count();
    let extreme_ratio = extreme_count as f64 / SAMPLE_SIZE as f64;
    assert!(
        extreme_ratio > 0.01,
        "queues Pareto trop légères : {extreme_ratio:.4}"
    );
}

// ============================================================================
// Tests de la distribution Uniforme
// ============================================================================

#[test]
fn uniform_mean_variance() {
    let config = DistributionConfig::uniform(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Uniforme(0,1) : moyenne = 0.5, variance = 1/12
    assert_mean(&samples, 0.5, 1.0 / 12.0_f64.sqrt());
    assert_variance(&samples, 1.0 / 12.0_f64.sqrt());
}

#[test]
fn uniform_quantiles() {
    let config = DistributionConfig::uniform(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Quantiles théoriques pour U(0,1)
    // Tolérance augmentée à 0.02 en raison de la variabilité statistique
    assert_quantile(&samples, 0.1, 0.10, 0.02); // 10%
    assert_quantile(&samples, 0.5, 0.50, 0.02); // médiane
    assert_quantile(&samples, 0.9, 0.90, 0.02); // 90%
}

#[test]
fn uniform_pdf_cdf() {
    let config = DistributionConfig::uniform(2.0, 5.0);
    let samples = generate_samples(&config, 1000);
    // Vérifier que les valeurs sont dans l'intervalle
    let all_in_range = samples.iter().all(|&x| (2.0..=5.0).contains(&x));
    assert!(
        all_in_range,
        "échantillons Uniform(2,5) hors de l'intervalle"
    );
}

#[test]
fn uniform_sample_range() {
    let config = DistributionConfig::uniform(-10.0, 10.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Vérifier que les valeurs sont dans [-10, 10]
    let min_val = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(min_val >= -10.0, "minimum trop petit : {min_val}");
    assert!(max_val <= 10.0, "maximum trop grand : {max_val}");
}

#[test]
fn uniform_deterministic() {
    let config = DistributionConfig::uniform(0.0, 1.0);
    let samples1 = generate_samples(&config, 100);
    let samples2 = generate_samples(&config, 100);
    assert_eq!(
        samples1, samples2,
        "même seed doit produire mêmes échantillons pour Uniform"
    );
}

// ============================================================================
// Tests du mélange (Mixture)
// ============================================================================

#[test]
fn mixture_mean_variance() {
    let config = DistributionConfig::mixture(vec![
        (0.5, DistributionConfig::normal(-1.0, 0.5)),
        (0.5, DistributionConfig::normal(1.0, 0.5)),
    ]);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Mélange symétrique : moyenne = 0
    assert_mean(&samples, 0.0, 0.5); // tolérance augmentée
                                     // Variance = E[X²] - E[X]² = 0.5*(1+0.25) + 0.5*(1+0.25) - 0 = 1.25
    let expected_var: f64 = 1.25;
    assert_variance(&samples, expected_var.sqrt());
}

#[test]
fn mixture_bimodal() {
    let config = DistributionConfig::mixture(vec![
        (0.5, DistributionConfig::normal(-2.0, 0.5)),
        (0.5, DistributionConfig::normal(2.0, 0.5)),
    ]);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Vérifier la bimodalité : beaucoup de valeurs loin de 0
    let far_from_zero = samples.iter().filter(|&&x| x.abs() > 1.5).count();
    let far_ratio = far_from_zero as f64 / SAMPLE_SIZE as f64;
    assert!(
        far_ratio > 0.4,
        "mélange bimodal doit avoir >40% de valeurs loin de zéro : {far_ratio:.4}"
    );
}

// ============================================================================
// Tests de comportement des queues
// ============================================================================

#[test]
fn normal_tail_decay() {
    let config = DistributionConfig::normal(0.0, 1.0);
    let samples = generate_samples(&config, SAMPLE_SIZE);
    // Pour N(0,1), P(|X| > 3) ≈ 0.0027
    let count_3sigma = samples.iter().filter(|&&x| x.abs() > 3.0).count();
    let ratio_3sigma = count_3sigma as f64 / SAMPLE_SIZE as f64;
    assert!(
        ratio_3sigma < 0.01,
        "queues normale trop lourdes : {ratio_3sigma:.4}"
    );
    assert!(
        ratio_3sigma > 0.001,
        "queues normale trop légères : {ratio_3sigma:.4}"
    );
}

#[test]
fn student_t_tail_heavier_than_normal() {
    let config_t = DistributionConfig::student_t(5.0);
    let config_normal = DistributionConfig::normal(0.0, 1.0);
    let samples_t = generate_samples(&config_t, SAMPLE_SIZE);
    let samples_normal = generate_samples(&config_normal, SAMPLE_SIZE);
    // Student-t doit avoir plus de valeurs extrêmes que la normale
    let count_t = samples_t.iter().filter(|&&x| x.abs() > 3.0).count();
    let count_normal = samples_normal.iter().filter(|&&x| x.abs() > 3.0).count();
    assert!(
        count_t > count_normal,
        "Student-t doit avoir plus de queues lourdes que la normale"
    );
}

// ============================================================================
// Tests de déterminisme
// ============================================================================

#[test]
fn distribution_deterministic_with_same_seed() {
    let config = DistributionConfig::normal(0.0, 1.0);
    let samples1 = generate_samples(&config, 100);
    let samples2 = generate_samples(&config, 100);
    assert_eq!(
        samples1, samples2,
        "même seed doit produire mêmes échantillons"
    );
}

// ============================================================================
// Tests de validation des paramètres
// ============================================================================

#[test]
fn invalid_parameters_rejected() {
    // Normale avec σ ≤ 0
    assert!(from_config(&DistributionConfig::normal(0.0, -1.0)).is_err());
    assert!(from_config(&DistributionConfig::normal(0.0, 0.0)).is_err());

    // Student-t avec df ≤ 0
    assert!(from_config(&DistributionConfig::student_t(0.0)).is_err());
    assert!(from_config(&DistributionConfig::student_t(-1.0)).is_err());

    // Pareto avec α ≤ 0
    assert!(from_config(&DistributionConfig::pareto(1.0, 0.0)).is_err());
    assert!(from_config(&DistributionConfig::pareto(1.0, -1.0)).is_err());

    // Weibull avec k ≤ 0
    assert!(from_config(&DistributionConfig::weibull(1.0, 0.0)).is_err());
    assert!(from_config(&DistributionConfig::weibull(1.0, -1.0)).is_err());
}

// ============================================================================
// Tests de propriétés générales
// ============================================================================

#[test]
fn distributions_produce_finite_values() {
    let configs = vec![
        DistributionConfig::normal(0.0, 1.0),
        DistributionConfig::student_t(5.0),
        DistributionConfig::laplace(0.0, 1.0),
        DistributionConfig::log_normal(0.0, 0.5),
        DistributionConfig::weibull(1.0, 2.0),
        DistributionConfig::pareto(1.0, 3.0),
        DistributionConfig::uniform(0.0, 1.0),
    ];

    for config in configs {
        let samples = generate_samples(&config, 1000);
        let all_finite = samples.iter().all(|x| x.is_finite());
        assert!(
            all_finite,
            "distribution {:?} doit produire uniquement des valeurs finies",
            config.kind
        );
    }
}
