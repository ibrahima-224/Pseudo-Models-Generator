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

//! Sous-module contenant les normes vectorielles et les statistiques de queues.

use crate::error::MathResult;

use super::basic_stats::{quantiles, std_population};

/// Calcule la norme L1 (somme des valeurs absolues).
pub fn norm_l1(data: &[f64]) -> MathResult<f64> {
    require_non_empty(data, "norm_l1")?;
    Ok(data.iter().map(|x| x.abs()).sum())
}

/// Calcule la norme L2 (racine carrée de la somme des carrés).
pub fn norm_l2(data: &[f64]) -> MathResult<f64> {
    require_non_empty(data, "norm_l2")?;
    Ok(data.iter().map(|x| x * x).sum::<f64>().sqrt())
}

/// Calcule la norme infinie (maximum des valeurs absolues).
pub fn norm_infinity(data: &[f64]) -> MathResult<f64> {
    require_non_empty(data, "norm_infinity")?;
    Ok(data.iter().map(|x| x.abs()).fold(0.0f64, f64::max))
}

/// Calcule les statistiques de queues (quantiles extrêmes).
pub fn tail_statistics(data: &[f64]) -> MathResult<(f64, f64, f64, f64)> {
    require_non_empty(data, "tail_statistics")?;
    let q = quantiles(data, &[0.99, 0.999, 0.9999])?;
    let sigma = std_population(data)?;
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let ratio = if sigma > 0.0 { max_abs / sigma } else { 0.0 };
    Ok((q[0], q[1], q[2], ratio))
}

/// Vérifie que la slice n'est pas vide.
fn require_non_empty(data: &[f64], func_name: &str) -> MathResult<()> {
    if data.is_empty() {
        Err(crate::error::MathError::EmptyData(format!(
            "{func_name} nécessite une slice non vide"
        )))
    } else {
        Ok(())
    }
}
