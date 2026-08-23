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

//! Tests d'intégration pour les politiques d'injection.
//!
//! Ce module teste la validation des politiques d'injection, la construction
//! de politiques adaptées à la profondeur, et les propriétés des politiques.

use pmg_injector::error::InjectorError;
use pmg_injector::injection_policy::{policy_for_depth, InjectionPolicy, LayerDepthProfile};

#[test]
fn default_policy_is_valid() {
    let policy = InjectionPolicy::default();
    assert!(policy.validate().is_ok());
}

#[test]
fn none_policy_is_valid() {
    let policy = InjectionPolicy::none();
    assert!(policy.validate().is_ok());
}

#[test]
fn custom_policy_is_valid() {
    let policy = InjectionPolicy::new(
        0.02, // outlier_frequency
        5.0,  // outlier_scale
        0.4,  // correlation_strength
        0.15, // low_rank_probability
        10,   // low_rank_rank
        0.3,  // low_rank_alpha
        0.1,  // heavy_tail_probability
        4.0,  // heavy_tail_df
        0.05, // sparse_structure_probability
        0.3,  // sparse_density
    )
    .unwrap();
    assert!(policy.validate().is_ok());
}

#[test]
fn outlier_frequency_out_of_bounds_rejected() {
    // Fréquence > 1 doit échouer
    assert!(matches!(
        InjectionPolicy::new(1.5, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
        Err(InjectorError::InvalidPolicy(_))
    ));

    // Fréquence négative doit échouer
    assert!(matches!(
        InjectionPolicy::new(-0.1, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
        Err(InjectorError::InvalidPolicy(_))
    ));
}

#[test]
fn outlier_scale_below_one_rejected() {
    // Un super-poids doit amplifier : scale < 1 rejeté
    assert!(matches!(
        InjectionPolicy::new(0.01, 0.5, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
        Err(InjectorError::InvalidPolicy(_))
    ));
}

#[test]
fn correlation_strength_one_rejected() {
    // ρ = 1 exclu (variance nulle de la composante indépendante)
    assert!(matches!(
        InjectionPolicy::new(0.01, 5.0, 1.0, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
        Err(InjectorError::InvalidPolicy(_))
    ));

    // ρ = 0 accepté
    assert!(InjectionPolicy::new(0.01, 5.0, 0.0, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5).is_ok());
}

#[test]
fn rank_zero_rejected() {
    assert!(matches!(
        InjectionPolicy::new(0.01, 5.0, 0.3, 0.2, 0, 0.5, 0.05, 4.0, 0.1, 0.5),
        Err(InjectorError::InvalidPolicy(_))
    ));
}

#[test]
fn sparse_density_zero_rejected() {
    assert!(matches!(
        InjectionPolicy::new(0.01, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.0),
        Err(InjectorError::InvalidPolicy(_))
    ));
}

#[test]
fn policy_for_depth_interpolates_correctly() {
    // Créer une politique de base avec un profil personnalisé
    let mut base = InjectionPolicy::default();
    base.depth_profile = LayerDepthProfile {
        probability_start: 1.0,
        probability_end: 2.0,
        amplitude_start: 1.0,
        amplitude_end: 3.0,
    };

    let first = policy_for_depth(&base, 0.0, 4).unwrap();
    let last = policy_for_depth(&base, 1.0, 4).unwrap();
    let mid = policy_for_depth(&base, 0.5, 4).unwrap();

    // Première couche : multiplicateur probability_start
    assert_eq!(first.outlier_frequency, base.outlier_frequency);

    // Dernière couche : multiplicateur probability_end
    assert_eq!(last.outlier_frequency, 2.0 * base.outlier_frequency);
    assert_eq!(last.outlier_scale, 3.0 * base.outlier_scale);

    // Milieu : interpolation linéaire
    assert!((mid.outlier_frequency - 1.5 * base.outlier_frequency).abs() < 1e-12);
    assert!((mid.outlier_scale - 2.0 * base.outlier_scale).abs() < 1e-12);
}

#[test]
fn policy_for_depth_rejects_invalid_input() {
    let base = InjectionPolicy::default();

    // Profondeur hors [0, 1]
    assert!(policy_for_depth(&base, -0.1, 4).is_err());
    assert!(policy_for_depth(&base, 1.1, 4).is_err());

    // Nombre de couches nul
    assert!(policy_for_depth(&base, 0.5, 0).is_err());
}

#[test]
fn layer_depth_profile_validation() {
    let mut profile = LayerDepthProfile::neutral();
    assert!(profile.validate().is_ok());

    // Valeur négative rejetée
    profile.amplitude_end = -1.0;
    assert!(profile.validate().is_err());

    // Valeur infinie rejetée
    profile.amplitude_end = f64::INFINITY;
    assert!(profile.validate().is_err());
}

#[test]
fn policy_serde_roundtrip() {
    let policy = InjectionPolicy::default();
    let json = serde_json::to_string(&policy).unwrap();
    let back: InjectionPolicy = serde_json::from_str(&json).unwrap();
    assert_eq!(policy, back);
}

#[test]
fn policy_none_disables_all_injections() {
    let policy = InjectionPolicy::none();
    assert_eq!(policy.outlier_frequency, 0.0);
    assert_eq!(policy.correlation_strength, 0.0);
    assert_eq!(policy.low_rank_probability, 0.0);
    assert_eq!(policy.heavy_tail_probability, 0.0);
    assert_eq!(policy.sparse_structure_probability, 0.0);
}

#[test]
fn policy_default_has_reasonable_values() {
    let policy = InjectionPolicy::default();

    // Vérifier que les valeurs par défaut sont raisonnables
    assert!(policy.outlier_frequency > 0.0 && policy.outlier_frequency < 1.0);
    assert!(policy.outlier_scale >= 1.0);
    assert!(policy.correlation_strength >= 0.0 && policy.correlation_strength < 1.0);
    assert!(policy.low_rank_probability >= 0.0 && policy.low_rank_probability <= 1.0);
    assert!(policy.low_rank_rank >= 1);
    assert!(policy.low_rank_alpha >= 0.0);
    assert!(policy.heavy_tail_probability >= 0.0 && policy.heavy_tail_probability <= 1.0);
    assert!(policy.heavy_tail_df > 0.0);
    assert!(
        policy.sparse_structure_probability >= 0.0 && policy.sparse_structure_probability <= 1.0
    );
    assert!(policy.sparse_density > 0.0 && policy.sparse_density <= 1.0);
}
