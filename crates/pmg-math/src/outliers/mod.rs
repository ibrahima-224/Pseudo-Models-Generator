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

//! Module des outliers — abstractions pour les super-poids et anomalies critiques.
//!
//! Ce module fournit les primitives mathématiques pour modéliser les anomalies
//! dans les tenseurs de poids, conformément au Sprint 9 du plan de développement.
//!
//! # Responsabilités
//!
//! - Modélisation des super-poids (additif ou multiplicatif) — [`model`];
//! - Calcul de l'amplitude des anomalies selon différentes stratégies — [`amplitude`];
//! - Politiques d'injection par couche — [`layer_policy`];
//!
//! # Conventions
//!
//! - Toutes les fonctions sont déterministes et reproductibles (utilisent `DeterministicRng`);
//! - Les paramètres sont validés et retournent des erreurs typées (`MathError`);
//! - La documentation est en français.

pub mod amplitude;
pub mod layer_policy;
pub mod model;

pub use amplitude::{compute_amplitude, AmplitudeStrategy};
pub use layer_policy::{layer_outlier_config, LayerPolicy};
pub use model::{OutlierModel, OutlierSpec};
