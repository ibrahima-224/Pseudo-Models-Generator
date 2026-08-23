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

//! Sous-module contenant les fonctions utilitaires pour l'injection de tenseurs.

use pmg_core::Shape;

use crate::injection_policy::InjectionPolicy;
use crate::sparse_structure::{BlockPattern, SparseStructureSpec};

/// Dimensions `(rows, cols)` si la shape est 2D, `None` sinon.
///
/// Les étapes structure/corrélation/bas-rang sont définies pour les matrices ;
/// un tenseur 1D (ou plus) ne reçoit que la distribution et les super-poids.
pub fn matrix_dims(shape: &Shape) -> Option<(usize, usize)> {
    if shape.rank() == 2 {
        let dims = shape.dims();
        Some((dims[0] as usize, dims[1] as usize))
    } else {
        None
    }
}

/// Construit la spécification sparse depuis la politique (bloc unique
/// d'environ 25 % de la matrice, densité du policy).
pub fn sparse_spec_from_policy(
    policy: &InjectionPolicy,
    rows: usize,
    cols: usize,
) -> SparseStructureSpec {
    let block_rows = (rows / 4).max(1);
    let block_cols = (cols / 4).max(1);
    SparseStructureSpec::new(
        BlockPattern::SingleBlock,
        policy.sparse_density,
        block_rows,
        block_cols,
        1,
        1,
    )
    .expect("spec sparse valide par construction")
}
