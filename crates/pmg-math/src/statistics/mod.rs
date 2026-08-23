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

//! Sous-module contenant les statistiques descriptives de base et les normes.
//!
//! Ce module fournit :
//! - Statistiques descriptives (somme, moyenne, variance, etc.)
//! - Normes vectorielles (L1, L2, infinie)
//! - Statistiques de queues (quantiles extrêmes)

// Sous-modules
mod basic_stats;
mod norms;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use basic_stats::{
    kurtosis, mean, median, min_max, quantiles, skewness, std_population, std_sample, sum, summary,
    variance_population, variance_sample, SummaryStats,
};
pub use norms::{norm_infinity, norm_l1, norm_l2, tail_statistics};

// Tests unitaires
#[cfg(test)]
mod tests;
