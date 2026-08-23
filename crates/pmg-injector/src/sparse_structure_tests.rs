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

//! Tests unitaires pour le module `sparse_structure`.

use super::{
    apply_sparse_mask, generate_sparse_structure, measured_density, BlockPattern,
    SparseStructureSpec,
};
use crate::error::InjectorError;
use pmg_math::rng::{derive_sub_seed, DeterministicRng};

/// Fonction utilitaire pour créer un générateur déterministe à partir d'une graine.
fn rng_for(seed: [u8; 32]) -> DeterministicRng {
    DeterministicRng::from_seed(derive_sub_seed(&seed, "sparse", 0))
}

/// Graine de base pour les tests déterministes.
fn base_seed() -> [u8; 32] {
    [23u8; 32]
}

/// Crée une spécification pour un bloc unique avec les paramètres donnés.
fn single_block(density: f64) -> SparseStructureSpec {
    SparseStructureSpec::new(BlockPattern::SingleBlock, density, 3, 4, 1, 1).unwrap()
}

#[test]
fn single_block_concentrates_nonzeros() {
    // 10×10, bloc 3×4 : au plus 12 non-nuls.
    let spec = single_block(0.12);
    let m = generate_sparse_structure(&mut rng_for(base_seed()), 10, 10, &spec, 1.0).unwrap();
    let nz = m.iter().filter(|&&x| x != 0.0).count();
    assert!(nz <= 12, "bloc trop grand : {nz} non-nuls");
    assert!(nz > 0, "bloc vide");
    let density = measured_density(&m, 0.0);
    assert!(density <= 0.12 + 1e-9);
}

#[test]
fn diagonal_band_produces_band_structure() {
    let spec = SparseStructureSpec::new(BlockPattern::DiagonalBand, 1.0, 1, 1, 1, 2).unwrap();
    let m = generate_sparse_structure(&mut rng_for(base_seed()), 6, 6, &spec, 1.0).unwrap();
    for row in 0..6usize {
        for col in 0..6usize {
            let in_band = col.abs_diff(row) <= 1;
            let nonzero = m[row * 6 + col] != 0.0;
            assert_eq!(nonzero, in_band, "position ({row},{col}) incohérente");
        }
    }
}

#[test]
fn grid_pattern_is_regular() {
    let spec = SparseStructureSpec::new(BlockPattern::Grid, 1.0, 2, 2, 3, 1).unwrap();
    let m = generate_sparse_structure(&mut rng_for(base_seed()), 9, 9, &spec, 1.0).unwrap();
    // Trois blocs de 2×2 sur la diagonale : positions (0,0),(0,1),…,
    // (3,3),(3,4),…, (6,6),(6,7).
    for row in 0..9 {
        for col in 0..9 {
            let in_block = matches!((row, col), (0..=1, 0..=1) | (3..=4, 3..=4) | (6..=7, 6..=7));
            assert_eq!(m[row * 9 + col] != 0.0, in_block);
        }
    }
}

#[test]
fn rows_and_columns_patterns() {
    // Densité 0.5 sur 8 lignes : pas de 2 → lignes paires.
    let spec = SparseStructureSpec::new(BlockPattern::Rows, 0.5, 1, 1, 1, 1).unwrap();
    let m = generate_sparse_structure(&mut rng_for(base_seed()), 8, 8, &spec, 1.0).unwrap();
    for row in 0..8 {
        for col in 0..8 {
            assert_eq!(m[row * 8 + col] != 0.0, row % 2 == 0);
        }
    }
    let spec = SparseStructureSpec::new(BlockPattern::Columns, 0.5, 1, 1, 1, 1).unwrap();
    let m = generate_sparse_structure(&mut rng_for(base_seed()), 8, 8, &spec, 1.0).unwrap();
    for row in 0..8 {
        for col in 0..8 {
            assert_eq!(m[row * 8 + col] != 0.0, col % 2 == 0);
        }
    }
}

#[test]
fn generation_is_deterministic() {
    let spec = single_block(0.2);
    let a = generate_sparse_structure(&mut rng_for(base_seed()), 20, 20, &spec, 1.0).unwrap();
    let b = generate_sparse_structure(&mut rng_for(base_seed()), 20, 20, &spec, 1.0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn invalid_spec_rejected() {
    assert!(SparseStructureSpec::new(BlockPattern::SingleBlock, 0.0, 2, 2, 1, 1).is_err());
    assert!(SparseStructureSpec::new(BlockPattern::SingleBlock, 1.5, 2, 2, 1, 1).is_err());
    assert!(SparseStructureSpec::new(BlockPattern::SingleBlock, 0.5, 0, 2, 1, 1).is_err());
    assert!(SparseStructureSpec::new(BlockPattern::Grid, 0.5, 2, 2, 0, 1).is_err());
    assert!(SparseStructureSpec::new(BlockPattern::DiagonalBand, 0.5, 1, 1, 1, 0).is_err());
}

#[test]
fn zero_dimensions_rejected() {
    let spec = single_block(0.5);
    assert!(matches!(
        generate_sparse_structure(&mut rng_for(base_seed()), 0, 5, &spec, 1.0),
        Err(InjectorError::InvalidTensor(_))
    ));
}

#[test]
fn apply_sparse_mask_zeroes_outside_structure() {
    let mut buf = vec![1.0f64; 36]; // 6×6
    let spec = SparseStructureSpec::new(BlockPattern::DiagonalBand, 1.0, 1, 1, 1, 1).unwrap();
    apply_sparse_mask(&mut buf, 6, 6, &spec).unwrap();
    for row in 0..6 {
        for col in 0..6 {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert_eq!(buf[row * 6 + col], expected);
        }
    }
}

#[test]
fn apply_sparse_mask_rejects_length_mismatch() {
    let spec = single_block(0.5);
    let mut buf = vec![1.0f64; 35];
    assert!(matches!(
        apply_sparse_mask(&mut buf, 6, 6, &spec),
        Err(InjectorError::InvalidTensor(_))
    ));
}

#[test]
fn measured_density_counts_nonzero() {
    assert_eq!(measured_density(&[0.0, 1.0, 0.0, 2.0], 1e-12), 0.5);
    assert_eq!(measured_density(&[], 1e-12), 0.0);
    // Avec un seuil ε, les petites valeurs comptent comme nulles.
    assert_eq!(measured_density(&[1e-15, 0.0], 1e-12), 0.0);
}
