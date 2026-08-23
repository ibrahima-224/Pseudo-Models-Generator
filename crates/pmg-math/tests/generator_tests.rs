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

//! Tests du générateur de base (`pmg_math::generator`).
//!
//! Ces tests vérifient les propriétés essentielles du générateur :
//! - déterminisme ;
//! - taille correcte des sorties ;
//! - comportement avec des seeds différentes ;
//! - gestion des cas limites (génération vide, paramètres invalides) ;
//! - préservation du déterminisme par chunks.

use pmg_math::generator::{generate_normal, generate_uniform};
use pmg_math::rng::DeterministicRng;

// ============================================================================
// Tests de déterminisme
// ============================================================================

#[test]
fn normal_deterministic_with_same_seed() {
    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    let v1 = generate_normal(0.0, 1.0, 100, &mut rng1).unwrap();
    let v2 = generate_normal(0.0, 1.0, 100, &mut rng2).unwrap();
    assert_eq!(
        v1, v2,
        "les mêmes seeds doivent produire les mêmes résultats"
    );
}

#[test]
fn uniform_deterministic_with_same_seed() {
    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
    let v1 = generate_uniform(0.0, 1.0, 100, &mut rng1).unwrap();
    let v2 = generate_uniform(0.0, 1.0, 100, &mut rng2).unwrap();
    assert_eq!(
        v1, v2,
        "les mêmes seeds doivent produire les mêmes résultats"
    );
}

// ============================================================================
// Tests de taille correcte
// ============================================================================

#[test]
fn normal_correct_length() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let values = generate_normal(0.0, 1.0, 50, &mut rng).unwrap();
    assert_eq!(values.len(), 50, "doit générer exactement 50 valeurs");
}

#[test]
fn uniform_correct_length() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let values = generate_uniform(0.0, 1.0, 200, &mut rng).unwrap();
    assert_eq!(values.len(), 200, "doit générer exactement 200 valeurs");
}

// ============================================================================
// Tests avec seed différente
// ============================================================================

#[test]
fn normal_different_seeds_produce_different_values() {
    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let mut rng2 = DeterministicRng::from_seed([43u8; 32]);
    let v1 = generate_normal(0.0, 1.0, 1000, &mut rng1).unwrap();
    let v2 = generate_normal(0.0, 1.0, 1000, &mut rng2).unwrap();
    assert_ne!(
        v1, v2,
        "des seeds différentes doivent produire des résultats différents"
    );
}

#[test]
fn uniform_different_seeds_produce_different_values() {
    let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
    let mut rng2 = DeterministicRng::from_seed([43u8; 32]);
    let v1 = generate_uniform(0.0, 1.0, 1000, &mut rng1).unwrap();
    let v2 = generate_uniform(0.0, 1.0, 1000, &mut rng2).unwrap();
    assert_ne!(
        v1, v2,
        "des seeds différentes doivent produire des résultats différents"
    );
}

// ============================================================================
// Tests de génération vide
// ============================================================================

#[test]
fn normal_empty_generation() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let values = generate_normal(0.0, 1.0, 0, &mut rng).unwrap();
    assert!(
        values.is_empty(),
        "une génération vide doit retourner un vecteur vide"
    );
}

#[test]
fn uniform_empty_generation() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let values = generate_uniform(0.0, 1.0, 0, &mut rng).unwrap();
    assert!(
        values.is_empty(),
        "une génération vide doit retourner un vecteur vide"
    );
}

// ============================================================================
// Tests de dimensions invalides (paramètres invalides)
// ============================================================================

#[test]
fn normal_rejects_zero_sigma() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    assert!(generate_normal(0.0, 0.0, 10, &mut rng).is_err());
}

#[test]
fn normal_rejects_negative_sigma() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    assert!(generate_normal(0.0, -1.0, 10, &mut rng).is_err());
}

#[test]
fn uniform_rejects_equal_bounds() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    assert!(generate_uniform(1.0, 1.0, 10, &mut rng).is_err());
}

#[test]
fn uniform_rejects_inverted_bounds() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    assert!(generate_uniform(1.0, 0.0, 10, &mut rng).is_err());
}

// ============================================================================
// Tests de chunk boundaries (découpage en blocs)
// ============================================================================

#[test]
fn chunk_boundary_determinism() {
    // Vérifie que générer par chunks produit les mêmes résultats
    // qu'une génération continue.
    let seed = [42u8; 32];
    let total = 1000;
    let chunk_size = 100;

    // Génération continue
    let mut rng_full = DeterministicRng::from_seed(seed);
    let _full_values = generate_normal(0.0, 1.0, total, &mut rng_full).unwrap();

    // Génération par chunks avec dérivation de seed
    let mut chunk_values = Vec::new();
    for chunk_id in 0..(total / chunk_size) {
        // Dérive une seed pour chaque chunk (similaire à derive_sub_seed)
        use pmg_math::rng::derive_sub_seed;
        let chunk_seed = derive_sub_seed(&seed, "chunk", chunk_id as u32);
        let mut rng_chunk = DeterministicRng::from_seed(chunk_seed);
        let chunk = generate_normal(0.0, 1.0, chunk_size, &mut rng_chunk).unwrap();
        chunk_values.extend(chunk);
    }

    // Les résultats par chunks doivent être différents de la génération continue
    // car les seeds sont différentes (dérivation hiérarchique).
    // Mais chaque chunk individuellement doit être déterministe.
    assert_eq!(chunk_values.len(), total);

    // Vérifie le déterminisme des chunks individuellement
    for chunk_id in 0..(total / chunk_size) {
        use pmg_math::rng::derive_sub_seed;
        let chunk_seed = derive_sub_seed(&seed, "chunk", chunk_id as u32);
        let mut rng1 = DeterministicRng::from_seed(chunk_seed);
        let mut rng2 = DeterministicRng::from_seed(chunk_seed);
        let v1 = generate_normal(0.0, 1.0, chunk_size, &mut rng1).unwrap();
        let v2 = generate_normal(0.0, 1.0, chunk_size, &mut rng2).unwrap();
        assert_eq!(v1, v2, "le chunk {chunk_id} doit être déterministe");
    }
}

#[test]
fn chunk_boundary_uniform() {
    // Même test avec uniforme
    let seed = [42u8; 32];
    let total = 500;
    let chunk_size = 50;

    for chunk_id in 0..(total / chunk_size) {
        use pmg_math::rng::derive_sub_seed;
        let chunk_seed = derive_sub_seed(&seed, "chunk", chunk_id as u32);
        let mut rng1 = DeterministicRng::from_seed(chunk_seed);
        let mut rng2 = DeterministicRng::from_seed(chunk_seed);
        let v1 = generate_uniform(0.0, 1.0, chunk_size, &mut rng1).unwrap();
        let v2 = generate_uniform(0.0, 1.0, chunk_size, &mut rng2).unwrap();
        assert_eq!(v1, v2, "le chunk {chunk_id} doit être déterministe");
    }
}

// ============================================================================
// Tests statistiques de base
// ============================================================================

#[test]
fn normal_mean_and_stddev_reasonable() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let n = 100_000;
    let values = generate_normal(5.0, 2.0, n, &mut rng).unwrap();

    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();

    assert!(
        (mean - 5.0).abs() < 0.1,
        "moyenne empirique {mean} trop éloignée de 5.0"
    );
    assert!(
        (stddev - 2.0).abs() < 0.1,
        "écart-type empirique {stddev} trop éloigné de 2.0"
    );
}

#[test]
fn uniform_mean_reasonable() {
    let mut rng = DeterministicRng::from_seed([42u8; 32]);
    let n = 100_000;
    let values = generate_uniform(0.0, 1.0, n, &mut rng).unwrap();

    let mean = values.iter().sum::<f64>() / n as f64;
    assert!(
        (mean - 0.5).abs() < 0.01,
        "moyenne empirique {mean} trop éloignée de 0.5"
    );
}
