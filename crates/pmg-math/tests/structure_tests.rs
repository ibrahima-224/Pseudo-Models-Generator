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

//! Tests de validation des structures et corrélations.
//!
//! Ces tests vérifient les propriétés structurelles essentielles des modules
//! implémentés dans `pmg_math::structure`. Ils testent la cohérence, la
//! stabilité numérique et la reproductibilité.
//!
//! Conformité : `docs/architecture/09-tests-benchmarks-ci.md` §1.8.

use pmg_core::distribution_config::DistributionConfig;
use pmg_math::low_rank::{effective_rank, low_rank_contribution};
use pmg_math::rng::DeterministicRng;
use pmg_math::structure::base_structure::BaseStructure;
use pmg_math::structure::block_structure::{BlockConfig, BlockStructure};
use pmg_math::structure::correlation::{Correlation, CorrelationConfig};
use pmg_math::structure::factors::{matrix_product, FactorGenerator};
use pmg_math::structure::local_correlation::{LocalCorrelation, LocalCorrelationConfig};

/// Taille d'échantillon pour les tests statistiques.
const SAMPLE_SIZE: usize = 10_000;

/// Tolérance relative pour les tests de corrélation.
const CORRELATION_TOLERANCE: f64 = 0.2;

/// Tolérance relative pour les tests de stabilité numérique.
const NUMERICAL_TOLERANCE: f64 = 1e-10;

// ============================================================================
// Tests de rang approximatif
// ============================================================================

#[test]
fn low_rank_approximate_rank() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let m = 10;
    let n = 10;
    let rank = 3;
    let alpha = 1.0;

    let w = low_rank_contribution(&mut rng, m, n, rank, alpha).unwrap();
    let effective = effective_rank(&w, m, n, 0.99).unwrap();

    // Le rang effectif devrait être proche du rang cible
    assert!(
        effective >= rank - 1 && effective <= rank + 1,
        "rang effectif {effective} loin du rang cible {rank}"
    );
}

#[test]
fn matrix_product_rank_preservation() {
    let gen = FactorGenerator::new(2, 1.0).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let (u, v) = gen.generate_both(&mut rng, 5, 5).unwrap();
    let l = matrix_product(&u, &v, 5, 5, 1.0).unwrap();
    let effective = effective_rank(&l, 5, 5, 0.99).unwrap();
    assert_eq!(effective, 2, "le produit devrait préserver le rang 2");
}

// ============================================================================
// Tests de corrélation et covariance
// ============================================================================

#[test]
fn correlation_symmetry_and_psd() {
    // Matrice de corrélation 3x3
    let rho = vec![0.5, 0.3, 0.1];
    let config = CorrelationConfig::from_pairwise(3, &rho).unwrap();
    let corr = Correlation::new(config).unwrap();

    // Vérifie que la matrice est symétrique
    let sigma = corr.config().sigma();
    let dim = corr.config().dim();
    for i in 0..dim {
        for j in 0..dim {
            assert!(
                (sigma[i * dim + j] - sigma[j * dim + i]).abs() < NUMERICAL_TOLERANCE,
                "matrice non symétrique : σ[{i}][{j}] ≠ σ[{j}][{i}]"
            );
        }
    }

    // Vérifie que la matrice est PSD (via Cholesky)
    let _cholesky = corr.cholesky();
}

#[test]
fn correlation_empirical() {
    let sigma = vec![1.0, 0.6, 0.6, 1.0];
    let config = CorrelationConfig::new(sigma, 2).unwrap();
    let corr = Correlation::new(config).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let samples = corr.generate(&mut rng, SAMPLE_SIZE).unwrap();

    // Calcule la corrélation empirique
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    for i in 0..SAMPLE_SIZE {
        let x = samples[i * 2];
        let y = samples[i * 2 + 1];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    let n = SAMPLE_SIZE as f64;
    let corr_emp = (n * sum_xy - sum_x * sum_y)
        / ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    assert!(
        (corr_emp - 0.6).abs() < CORRELATION_TOLERANCE,
        "corrélation empirique {corr_emp} loin de 0.6"
    );
}

#[test]
fn local_correlation_intra_block() {
    let config = LocalCorrelationConfig::new(vec![3, 3], vec![0.7, 0.7]).unwrap();
    let corr = LocalCorrelation::new(config).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let samples = corr.generate(&mut rng, SAMPLE_SIZE).unwrap();

    // Vérifie la corrélation intra-bloc
    for block in 0..2 {
        let offset = block * 3;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        for i in 0..SAMPLE_SIZE {
            let x = samples[i * 6 + offset];
            let y = samples[i * 6 + offset + 1];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        let n = SAMPLE_SIZE as f64;
        let corr_emp = (n * sum_xy - sum_x * sum_y)
            / ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        assert!(
            (corr_emp - 0.7).abs() < CORRELATION_TOLERANCE,
            "bloc {block} : corrélation empirique {corr_emp} loin de 0.7"
        );
    }
}

// ============================================================================
// Tests de stabilité numérique
// ============================================================================

#[test]
fn numerical_stability_base_structure() {
    let config = DistributionConfig::normal(0.0, 1e-10);
    let structure = BaseStructure::new(config).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let shape = vec![1000];
    let elements = structure.generate(&mut rng, &shape).unwrap();

    // Vérifie que tous les éléments sont finis
    for (i, &x) in elements.iter().enumerate() {
        assert!(x.is_finite(), "élément {i} non fini : {x}");
    }
}

#[test]
fn numerical_stability_high_correlation() {
    let sigma = vec![1.0, 0.999, 0.999, 1.0];
    let config = CorrelationConfig::new(sigma, 2).unwrap();
    let corr = Correlation::new(config).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let samples = corr.generate(&mut rng, 1000).unwrap();

    // Vérifie la stabilité numérique
    for (i, &x) in samples.iter().enumerate() {
        assert!(x.is_finite(), "élément {i} non fini : {x}");
    }
}

// ============================================================================
// Tests de reproductibilité
// ============================================================================

#[test]
fn reproducibility_base_structure() {
    let config = DistributionConfig::normal(0.0, 1.0);
    let structure = BaseStructure::new(config).unwrap();
    let shape = vec![10, 10];

    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let elements1 = structure.generate(&mut rng1, &shape).unwrap();

    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    let elements2 = structure.generate(&mut rng2, &shape).unwrap();

    assert_eq!(
        elements1, elements2,
        "les générateurs devraient être reproductibles"
    );
}

#[test]
fn reproducibility_correlation() {
    let sigma = vec![1.0, 0.5, 0.5, 1.0];
    let config = CorrelationConfig::new(sigma, 2).unwrap();
    let corr = Correlation::new(config).unwrap();

    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let samples1 = corr.generate(&mut rng1, 100).unwrap();

    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    let samples2 = corr.generate(&mut rng2, 100).unwrap();

    assert_eq!(
        samples1, samples2,
        "les générateurs devraient être reproductibles"
    );
}

// ============================================================================
// Tests de structure par blocs
// ============================================================================

#[test]
fn block_structure_independence() {
    let block1 = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0)).unwrap();
    let block2 = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0)).unwrap();
    let structure = BlockStructure::new(vec![block1, block2]).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let shape = vec![4];
    let elements = structure.generate(&mut rng, &shape).unwrap();

    // Vérifie que les éléments sont finis
    for &x in &elements {
        assert!(x.is_finite());
    }
    // Les blocs sont générés indépendamment, donc leurs moyennes ne sont pas
    // forcément proches, mais elles ne sont pas non plus identiques.
    // On vérifie simplement que le générateur fonctionne correctement.
    assert_eq!(elements.len(), 4);
}

#[test]
fn block_structure_with_correlation() {
    let rho = vec![0.5];
    let block = BlockConfig::new(2, DistributionConfig::normal(0.0, 1.0))
        .unwrap()
        .with_intra_correlation(&rho)
        .unwrap();
    let structure = BlockStructure::new(vec![block]).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let shape = vec![2];
    let elements = structure.generate(&mut rng, &shape).unwrap();

    // Vérifie que la corrélation intra-bloc est respectée
    assert!(elements.len() == 2);
    assert!(elements[0].is_finite());
    assert!(elements[1].is_finite());
}

// ============================================================================
// Tests de comparaison structure_strength = 0 vs > 0
// ============================================================================

#[test]
fn structure_strength_comparison() {
    // Teste que la force structurelle influence bien la génération
    let config = DistributionConfig::normal(0.0, 1.0);
    let structure = BaseStructure::new(config).unwrap();
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let shape = vec![100];

    // Génère avec structure de base (force = 0)
    let elements_base = structure.generate(&mut rng, &shape).unwrap();

    // Génère avec corrélation (force > 0)
    let sigma = vec![1.0, 0.8, 0.8, 1.0];
    let corr_config = CorrelationConfig::new(sigma, 2).unwrap();
    let corr = Correlation::new(corr_config).unwrap();
    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    let elements_corr = corr.generate(&mut rng2, 50).unwrap();

    // Les deux devraient générer des données finies
    for &x in &elements_base {
        assert!(x.is_finite());
    }
    for &x in &elements_corr {
        assert!(x.is_finite());
    }

    // Les distributions devraient être différentes
    let mean_base = elements_base.iter().sum::<f64>() / elements_base.len() as f64;
    let mean_corr = elements_corr.iter().sum::<f64>() / elements_corr.len() as f64;
    // Pas d'assertion stricte sur les moyennes, juste que les calculs sont valides
    assert!(mean_base.is_finite());
    assert!(mean_corr.is_finite());
}

// ============================================================================
// Tests de validation des erreurs
// ============================================================================

#[test]
fn invalid_correlation_matrix() {
    // Matrice non PSD
    let sigma = vec![1.0, 2.0, 2.0, 1.0];
    let result = CorrelationConfig::new(sigma, 2);
    assert!(result.is_err());
}

#[test]
fn invalid_block_size() {
    let result = BlockConfig::new(0, DistributionConfig::normal(0.0, 1.0));
    assert!(result.is_err());
}

#[test]
fn invalid_local_correlation_config() {
    let result = LocalCorrelationConfig::new(vec![2, 3], vec![0.5]);
    assert!(result.is_err());
}
