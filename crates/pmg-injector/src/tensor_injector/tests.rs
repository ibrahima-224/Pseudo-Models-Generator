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

//! Tests unitaires pour le module tensor_injector.

use super::*;
use crate::distribution_from_family;
use crate::injection_policy::InjectionPolicy;
use pmg_blueprint::tensor_spec::{DistributionFamily, TensorSpec};
use pmg_core::{DType, Shape, TensorRole};
use pmg_math::rng::{DeterministicRng, SeedPlan};

fn plan() -> SeedPlan<'static> {
    SeedPlan {
        seed_global: 42,
        model_id: "glm-5.2",
        tensor_name: "model.layers.3.mlp.gate.weight",
        layer_id: Some(3),
        generation_version: "1.0.0",
    }
}

fn spec_2d() -> TensorSpec {
    TensorSpec::new(
        "model.layers.3.mlp.gate.weight",
        Shape::new(vec![64, 32]).unwrap(),
        DType::F32,
        TensorRole::Other,
    )
    .unwrap()
}

#[test]
fn canonical_order_is_explicit() {
    // L'ordre canonique est matérialisé et testé (spécification §4.8).
    assert_eq!(
        InjectionStage::ORDER,
        [
            InjectionStage::Distribution,
            InjectionStage::Structure,
            InjectionStage::Correlation,
            InjectionStage::LowRank,
            InjectionStage::SuperWeights,
        ]
    );
    // Domaines de seed tous distincts (séparation des flux).
    let domains: Vec<&str> = InjectionStage::ORDER.iter().map(|s| s.domain()).collect();
    let mut sorted = domains.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        5,
        "domaines de seed non distincts : {domains:?}"
    );
}

#[test]
fn injection_is_deterministic() {
    let a = TensorInjector::from_seed_plan(&spec_2d(), InjectionPolicy::default(), &plan());
    let b = TensorInjector::from_seed_plan(&spec_2d(), InjectionPolicy::default(), &plan());
    let va = a.inject().unwrap();
    let vb = b.inject().unwrap();
    assert_eq!(va, vb, "le tenseur doit être identique bit à bit");
    assert_eq!(va.len(), 64 * 32);
    assert!(va.iter().all(|x| x.is_finite()));
}

#[test]
fn different_seed_different_tensor() {
    let a = TensorInjector::from_seed_plan(&spec_2d(), InjectionPolicy::default(), &plan());
    let other = SeedPlan {
        seed_global: 43,
        ..plan()
    };
    let b = TensorInjector::from_seed_plan(&spec_2d(), InjectionPolicy::default(), &other);
    assert_ne!(a.inject().unwrap(), b.inject().unwrap());
}

#[test]
fn none_policy_leaves_base_tensor_untouched() {
    // apply_to avec une politique neutre : le buffer est inchangé.
    let injector = TensorInjector::from_seed_plan(&spec_2d(), InjectionPolicy::none(), &plan());
    let mut buf = vec![1.0f64; 64 * 32];
    let before = buf.clone();
    injector.apply_to(&mut buf).unwrap();
    assert_eq!(buf, before);
}

#[test]
fn super_weights_only_policy_produces_extremes() {
    let mut policy = InjectionPolicy::none();
    policy.outlier_frequency = 0.05;
    policy.outlier_scale = 10.0;
    let injector = TensorInjector::from_seed_plan(&spec_2d(), policy, &plan());
    let out = injector.inject().unwrap();
    // Le tenseur de base est N(0,1) : |x| > 10 est quasi impossible sans
    // injection. Les valeurs extrêmes attestent des super-poids.
    let extremes = out.iter().filter(|&&x| x.abs() > 10.0).count();
    assert!(extremes > 0, "aucun super-poids détecté");
}

#[test]
fn correlation_policy_raises_empirical_correlation() {
    let mut policy = InjectionPolicy::none();
    policy.correlation_strength = 0.8;
    let injector = TensorInjector::from_seed_plan(&spec_2d(), policy, &plan());
    let out = injector.inject().unwrap();
    let rows = 64;
    let cols = 32;
    // Corrélation empirique moyenne entre colonnes ≈ ρ² = 0.64.
    let c = crate::correlated::empirical_correlation(&out, rows, cols).unwrap();
    let mut sum = 0.0;
    let mut n = 0;
    for a in 0..cols {
        for b in (a + 1)..cols {
            sum += c[a * cols + b];
            n += 1;
        }
    }
    let mean_offdiag = sum / n as f64;
    assert!(
        (mean_offdiag - 0.64).abs() < 0.1,
        "corrélation moyenne hors tolérance : {mean_offdiag}"
    );
}

#[test]
fn one_d_tensor_gets_distribution_and_outliers_only() {
    // Shape 1D : les étapes 2D sont ignorées, le tenseur reste valide.
    let spec = TensorSpec::new(
        "model.embed_tokens.weight",
        Shape::new(vec![1024]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();
    let injector = TensorInjector::from_seed_plan(&spec, InjectionPolicy::default(), &plan());
    let out = injector.inject().unwrap();
    assert_eq!(out.len(), 1024);
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn apply_to_rejects_length_mismatch() {
    let injector = TensorInjector::from_seed_plan(&spec_2d(), InjectionPolicy::none(), &plan());
    let mut buf = vec![0.0f64; 10];
    assert!(injector.apply_to(&mut buf).is_err());
}

#[test]
fn matrix_dims_detects_2d_only() {
    let s2 = Shape::new(vec![4, 4]).unwrap();
    assert_eq!(matrix_dims(&s2), Some((4, 4)));
    let s1 = Shape::new(vec![4]).unwrap();
    assert_eq!(matrix_dims(&s1), None);
}

#[test]
fn distribution_family_mappings_produce_expected_moments() {
    // Normal : moyenne et écart-type exacts.
    let mut rng = DeterministicRng::from_seed([1u8; 32]);
    let mut d = distribution_from_family(DistributionFamily::Normal, 0.0, 1.0).unwrap();
    let samples: Vec<f64> = (0..100_000).map(|_| d.sample(&mut rng)).collect();
    let mean = pmg_math::statistics::mean(&samples).unwrap();
    let std = pmg_math::statistics::std_population(&samples).unwrap();
    assert!(mean.abs() < 0.02, "moyenne {mean}");
    assert!((std - 1.0).abs() < 0.02, "std {std}");

    // Mixture bimodal : variance exacte σ².
    let mut rng = DeterministicRng::from_seed([2u8; 32]);
    let mut d = distribution_from_family(DistributionFamily::Mixture, 1.0, 2.0).unwrap();
    let samples: Vec<f64> = (0..100_000).map(|_| d.sample(&mut rng)).collect();
    let mean = pmg_math::statistics::mean(&samples).unwrap();
    let std = pmg_math::statistics::std_population(&samples).unwrap();
    assert!((mean - 1.0).abs() < 0.05, "moyenne {mean}");
    assert!((std - 2.0).abs() < 0.05, "std {std}");

    // Uniform : variance exacte σ².
    let mut rng = DeterministicRng::from_seed([3u8; 32]);
    let mut d = distribution_from_family(DistributionFamily::Uniform, 0.0, 3.0).unwrap();
    let samples: Vec<f64> = (0..100_000).map(|_| d.sample(&mut rng)).collect();
    let std = pmg_math::statistics::std_population(&samples).unwrap();
    assert!((std - 3.0).abs() < 0.05, "std {std}");
}

#[test]
fn distribution_family_rejects_invalid_stddev() {
    assert!(distribution_from_family(DistributionFamily::Normal, 0.0, 0.0).is_err());
    assert!(distribution_from_family(DistributionFamily::LogNormal, 0.0, 1.0).is_err());
}
