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

//! Tests des super-poids et anomalies critiques (Sprint 9).
//!
//! Ces tests vérifient les propriétés essentielles des primitives d'outliers :
//! - nombre d'outliers ;
//! - position des outliers ;
//! - amplitude des outliers ;
//! - déterminisme ;
//! - signe des anomalies ;
//! - distribution des valeurs ;
//! - invariant fondamental : fréquence nulle → pas de modification.

use pmg_math::outliers::amplitude::{compute_amplitude, AmplitudeStrategy};
use pmg_math::outliers::layer_policy::{LayerOutlierConfig, LayerPolicy};
use pmg_math::outliers::model::{OutlierModel, OutlierSpec, OutlierStrategy};
use pmg_math::rng::DeterministicRng;

// ============================================================================
// Tests du nombre d'outliers
// ============================================================================

#[test]
fn outlier_count_matches_mask() {
    let data = vec![0.0; 100];
    let mask = vec![false; 100];
    let specs = vec![];

    let model = OutlierModel::new(OutlierStrategy::Additive);
    let mut buffer = data.clone();
    assert!(model.apply_to_buffer(&mut buffer, &mask, &specs).is_ok());
    // Aucun outlier, le buffer doit être inchangé.
    assert_eq!(buffer, data);
}

#[test]
fn outlier_count_with_some_outliers() {
    let mask = vec![true, false, true, false, true];
    let specs = vec![
        OutlierSpec::new(OutlierStrategy::Additive, 1.0, true).unwrap(),
        OutlierSpec::new(OutlierStrategy::Additive, 2.0, true).unwrap(),
        OutlierSpec::new(OutlierStrategy::Additive, 3.0, true).unwrap(),
    ];
    let model = OutlierModel::new(OutlierStrategy::Additive);
    let mut buffer = vec![0.0; 5];
    assert!(model.apply_to_buffer(&mut buffer, &mask, &specs).is_ok());
    // Les valeurs aux positions 0, 2, 4 doivent être modifiées.
    assert_eq!(buffer[0], 1.0);
    assert_eq!(buffer[1], 0.0); // inchangé
    assert_eq!(buffer[2], 2.0);
    assert_eq!(buffer[3], 0.0); // inchangé
    assert_eq!(buffer[4], 3.0);
}

// ============================================================================
// Tests de position
// ============================================================================

#[test]
fn outliers_appear_at_correct_positions() {
    let mask = vec![false, true, false, true, false];
    let specs = vec![
        OutlierSpec::new(OutlierStrategy::Additive, 10.0, true).unwrap(),
        OutlierSpec::new(OutlierStrategy::Additive, 20.0, true).unwrap(),
    ];
    let model = OutlierModel::new(OutlierStrategy::Additive);
    let mut buffer = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(model.apply_to_buffer(&mut buffer, &mask, &specs).is_ok());
    assert_eq!(buffer[0], 1.0); // inchangé
    assert_eq!(buffer[1], 12.0); // 2 + 10
    assert_eq!(buffer[2], 3.0); // inchangé
    assert_eq!(buffer[3], 24.0); // 4 + 20
    assert_eq!(buffer[4], 5.0); // inchangé
}

// ============================================================================
// Tests d'amplitude
// ============================================================================

#[test]
fn amplitude_fixed() {
    let strategy = AmplitudeStrategy::Fixed(5.0);
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
    assert_eq!(amp, 5.0);
}

#[test]
fn amplitude_relative_to_std() {
    let strategy = AmplitudeStrategy::RelativeToStd { k: 2.0 };
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let amp = compute_amplitude(&strategy, &data, &mut rng).unwrap();
    // L'écart-type de [1,2,3,4,5] est ~1.5811
    let std = pmg_math::statistics::std_sample(&data).unwrap();
    assert!((amp - 2.0 * std).abs() < 1e-10);
}

#[test]
fn amplitude_quantile_based() {
    let strategy = AmplitudeStrategy::QuantileBased { p: 0.5 };
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let amp = compute_amplitude(&strategy, &data, &mut rng).unwrap();
    // Le quantile médian est 3.0
    assert!((amp - 3.0).abs() < 1e-10);
}

// ============================================================================
// Tests de déterminisme
// ============================================================================

#[test]
fn deterministic_injection_with_same_seed() {
    let mask = vec![true, false, true];
    let specs = vec![
        OutlierSpec::new(OutlierStrategy::Additive, 1.0, true).unwrap(),
        OutlierSpec::new(OutlierStrategy::Additive, 2.0, true).unwrap(),
    ];

    let model = OutlierModel::new(OutlierStrategy::Additive);
    let mut buffer1 = vec![0.0; 3];
    let mut buffer2 = vec![0.0; 3];
    assert!(model.apply_to_buffer(&mut buffer1, &mask, &specs).is_ok());
    assert!(model.apply_to_buffer(&mut buffer2, &mask, &specs).is_ok());
    assert_eq!(buffer1, buffer2, "même masque et specs → même résultat");
}

#[test]
fn deterministic_amplitude_with_same_seed() {
    let strategy = AmplitudeStrategy::HeavyTail { df: 3.0 };
    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    let amp1 = compute_amplitude(&strategy, &[], &mut rng1).unwrap();
    let amp2 = compute_amplitude(&strategy, &[], &mut rng2).unwrap();
    assert_eq!(amp1, amp2, "même seed → même amplitude");
}

// ============================================================================
// Tests de signe
// ============================================================================

#[test]
fn positive_outlier_increases_value() {
    let model = OutlierModel::new(OutlierStrategy::Additive);
    let spec = OutlierSpec::new(OutlierStrategy::Additive, 5.0, true).unwrap();
    let result = model.apply(10.0, &spec);
    assert_eq!(result, 15.0);
}

#[test]
fn negative_outlier_decreases_value() {
    let model = OutlierModel::new(OutlierStrategy::Additive);
    let spec = OutlierSpec::new(OutlierStrategy::Additive, 5.0, false).unwrap();
    let result = model.apply(10.0, &spec);
    assert_eq!(result, 5.0);
}

#[test]
fn multiplicative_positive_outlier() {
    let model = OutlierModel::new(OutlierStrategy::Multiplicative);
    let spec = OutlierSpec::new(OutlierStrategy::Multiplicative, 0.5, true).unwrap();
    let result = model.apply(10.0, &spec);
    assert_eq!(result, 15.0); // 10 * 1.5
}

#[test]
fn multiplicative_negative_outlier() {
    let model = OutlierModel::new(OutlierStrategy::Multiplicative);
    let spec = OutlierSpec::new(OutlierStrategy::Multiplicative, 0.5, false).unwrap();
    let result = model.apply(10.0, &spec);
    assert_eq!(result, 5.0); // 10 * 0.5
}

// ============================================================================
// Tests de distribution
// ============================================================================

#[test]
fn outlier_amplitude_is_finite() {
    let strategy = AmplitudeStrategy::HeavyTail { df: 2.0 };
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    for _ in 0..100 {
        let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
        assert!(amp.is_finite(), "amplitude doit être finie");
        assert!(amp >= 0.0, "amplitude doit être ≥ 0");
    }
}

#[test]
fn layer_policy_config_distribution() {
    let mut policy = LayerPolicy::new();
    policy.add_rule(0, 10, LayerOutlierConfig::low()).unwrap();
    policy
        .add_rule(11, 20, LayerOutlierConfig::medium())
        .unwrap();
    policy.add_rule(21, 30, LayerOutlierConfig::high()).unwrap();

    // Vérifie que chaque plage a la bonne probabilité.
    for i in 0..=10 {
        let config = policy.config_for_layer(i).unwrap();
        assert_eq!(config.probability, 0.01);
    }
    for i in 11..=20 {
        let config = policy.config_for_layer(i).unwrap();
        assert_eq!(config.probability, 0.05);
    }
    for i in 21..=30 {
        let config = policy.config_for_layer(i).unwrap();
        assert_eq!(config.probability, 0.10);
    }
    assert!(policy.config_for_layer(31).is_none());
}

// ============================================================================
// Test essentiel : fréquence nulle → pas de modification
// ============================================================================

#[test]
fn zero_frequency_no_modification() {
    // Simule un masque vide (aucun outlier).
    let mask = vec![false; 100];
    let specs = vec![];
    let original = (0..100).map(|i| i as f64).collect::<Vec<_>>();
    let mut buffer = original.clone();

    let model = OutlierModel::new(OutlierStrategy::Additive);
    assert!(model.apply_to_buffer(&mut buffer, &mask, &specs).is_ok());
    assert_eq!(
        buffer, original,
        "fréquence nulle doit produire le même résultat que sans injection"
    );
}

#[test]
fn zero_frequency_with_model_changes_nothing() {
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut buffer = original.clone();
    let mask = vec![false; 5];
    let specs = vec![];

    let model = OutlierModel::new(OutlierStrategy::Multiplicative);
    assert!(model.apply_to_buffer(&mut buffer, &mask, &specs).is_ok());
    assert_eq!(buffer, original);
}
