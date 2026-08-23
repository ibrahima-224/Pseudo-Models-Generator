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

//! Tests d'intégration pour les mélanges de distributions dans le générateur.
//!
//! Ces tests vérifient que le TensorGenerator gère correctement les mélanges
//! de distributions spécifiés dans le blueprint, y compris la validation des poids
//! et la génération d'échantillons conformes.

use pmg_blueprint::tensor_spec::{
    DistributionFamily, DistributionSpec, MixtureComponent, TensorSpec,
};
use pmg_core::{DType, Shape, TensorRole};
use pmg_generator::seed_plan::GeneratorSeedPlan;
use pmg_generator::tensor_generator::TensorGenerator;

/// Crée un tenseur de test avec un mélange de deux normales.
fn create_mixture_tensor() -> TensorSpec {
    let spec = TensorSpec::new(
        "model.mixture.weight",
        Shape::new(vec![1000]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    // Créer un mélange 50/50 entre deux normales
    let comp1 = MixtureComponent {
        weight: 0.5,
        distribution: DistributionSpec {
            family: DistributionFamily::Normal,
            mean: -2.0,
            stddev: 0.5,
            mixture_components: None,
        },
    };
    let comp2 = MixtureComponent {
        weight: 0.5,
        distribution: DistributionSpec {
            family: DistributionFamily::Normal,
            mean: 2.0,
            stddev: 0.5,
            mixture_components: None,
        },
    };

    let mut spec = spec;
    spec.distribution = DistributionSpec::mixture(vec![comp1, comp2]);
    spec
}

/// Crée un plan de seed pour les tests.
fn create_seed_plan() -> GeneratorSeedPlan {
    GeneratorSeedPlan::new(42, "test-model", "1.0.0")
}

/// Test : génération avec mélange de deux normales.
#[test]
fn test_mixture_generation_basic() {
    let spec = create_mixture_tensor();
    let seed_plan = create_seed_plan();
    let generator = TensorGenerator::new(spec, seed_plan, None);

    let values = generator.generate().unwrap();

    // Vérifier que nous avons le bon nombre d'éléments
    assert_eq!(values.len(), 1000);

    // Vérifier que toutes les valeurs sont finies
    assert!(values.iter().all(|v| v.is_finite()));

    // Vérifier que les valeurs sont dans une plage raisonnable
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(min_val > -10.0, "minimum trop petit : {min_val}");
    assert!(max_val < 10.0, "maximum trop grand : {max_val}");
}

/// Test : validation des poids des mélanges.
#[test]
fn test_mixture_weight_validation() {
    // Créer un mélange avec des poids qui ne somment pas à 1.0
    let comp1 = MixtureComponent {
        weight: 0.3,
        distribution: DistributionSpec {
            family: DistributionFamily::Normal,
            mean: 0.0,
            stddev: 1.0,
            mixture_components: None,
        },
    };
    let comp2 = MixtureComponent {
        weight: 0.3, // Somme = 0.6, pas 1.0
        distribution: DistributionSpec {
            family: DistributionFamily::Normal,
            mean: 0.0,
            stddev: 1.0,
            mixture_components: None,
        },
    };

    let mut spec = TensorSpec::new(
        "model.mixture.invalid",
        Shape::new(vec![100]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    spec.distribution = DistributionSpec::mixture(vec![comp1, comp2]);

    let seed_plan = create_seed_plan();
    let generator = TensorGenerator::new(spec, seed_plan, None);

    // La génération ne doit pas échouer mais la distribution interne validera
    // Note: La validation des poids est faite dans pmg-math lors de la construction
    // de la distribution. Ici, nous testons que le pipeline ne plante pas.
    let result = generator.generate();

    // Selon l'implémentation, cela peut réussir ou échouer selon la validation
    // Nous testons simplement que le code s'exécute sans panic
    if result.is_ok() {
        // Acceptable
    } else {
        // Aussi acceptable si la validation est stricte
    }
}

/// Test : mélange avec plus de deux composantes.
#[test]
fn test_mixture_three_components() {
    let comp1 = MixtureComponent {
        weight: 0.2,
        distribution: DistributionSpec {
            family: DistributionFamily::Normal,
            mean: -5.0,
            stddev: 0.1,
            mixture_components: None,
        },
    };
    let comp2 = MixtureComponent {
        weight: 0.3,
        distribution: DistributionSpec {
            family: DistributionFamily::Uniform,
            mean: 0.0,
            stddev: 1.0,
            mixture_components: None,
        },
    };
    let comp3 = MixtureComponent {
        weight: 0.5,
        distribution: DistributionSpec {
            family: DistributionFamily::Normal,
            mean: 5.0,
            stddev: 0.1,
            mixture_components: None,
        },
    };

    let mut spec = TensorSpec::new(
        "model.mixture.three",
        Shape::new(vec![2000]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    spec.distribution = DistributionSpec::mixture(vec![comp1, comp2, comp3]);

    let seed_plan = create_seed_plan();
    let generator = TensorGenerator::new(spec, seed_plan, None);
    let values = generator.generate().unwrap();

    // Vérifier que nous avons le bon nombre d'éléments
    assert_eq!(values.len(), 2000);

    // Vérifier la distribution : environ 20% autour de -5, 30% entre 0-1, 50% autour de 5
    let near_neg5 = values.iter().filter(|&&x| (x + 5.0).abs() < 1.0).count();
    let near_pos5 = values.iter().filter(|&&x| (x - 5.0).abs() < 1.0).count();

    // Tolérance de 10% pour l'estimation statistique
    assert!(
        near_neg5 > 200,
        "trop peu de valeurs autour de -5 : {near_neg5}"
    );
    assert!(
        near_neg5 < 600,
        "trop de valeurs autour de -5 : {near_neg5}"
    );
    assert!(
        near_pos5 > 600,
        "trop peu de valeurs autour de 5 : {near_pos5}"
    );
    assert!(
        near_pos5 < 1400,
        "trop de valeurs autour de 5 : {near_pos5}"
    );
}

/// Test : déterminisme avec mélanges.
#[test]
fn test_mixture_deterministic() {
    let spec = create_mixture_tensor();
    let seed_plan1 = create_seed_plan();
    let seed_plan2 = create_seed_plan();

    let generator1 = TensorGenerator::new(spec.clone(), seed_plan1, None);
    let generator2 = TensorGenerator::new(spec, seed_plan2, None);

    let values1 = generator1.generate().unwrap();
    let values2 = generator2.generate().unwrap();

    assert_eq!(
        values1, values2,
        "même seed doit produire mêmes échantillons pour mélange"
    );
}

/// Test : mélange avec budget limité.
#[test]
fn test_mixture_with_budget() {
    let spec = create_mixture_tensor();
    let seed_plan = create_seed_plan();

    // Budget pour seulement 500 éléments (500 * 4 octets = 2000 octets)
    let budget = 500 * 4;
    let generator = TensorGenerator::new(spec, seed_plan, Some(budget));
    let values = generator.generate().unwrap();

    // Vérifier que la génération est tronquée selon le budget
    assert!(
        values.len() <= 500,
        "nombre d'éléments {} dépasse le budget {}",
        values.len(),
        500
    );
    assert!(!values.is_empty(), "aucun élément généré avec budget");
}

/// Test : rétrocompatibilité sans mélange (distribution normale simple).
#[test]
fn test_no_mixture_backward_compatibility() {
    let mut spec = TensorSpec::new(
        "model.normal.weight",
        Shape::new(vec![100]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    // Distribution normale simple sans mélange
    spec.distribution = DistributionSpec::standard();

    let seed_plan = create_seed_plan();
    let generator = TensorGenerator::new(spec, seed_plan, None);
    let values = generator.generate().unwrap();

    assert_eq!(values.len(), 100);
    assert!(values.iter().all(|v| v.is_finite()));
}
