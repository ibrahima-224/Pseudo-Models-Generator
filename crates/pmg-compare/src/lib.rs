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

//! Crate `pmg-compare` — comparaison **metadata-only** de modèles.
//!
//! Compare deux modèles sur leurs métadonnées uniquement (configs, architecture,
//! tenseurs, shapes, dtypes, sharding) **sans jamais lire les poids**. La sortie
//! affiche obligatoirement `Poids : NON COMPARÉS` / `Comparaison metadata-only`.
//!
//! ## Responsabilité
//!
//! - comparaison des configurations, architectures, tenseurs, shapes, dtypes,
//!   shards ;
//! - diff structurée et score de similarité ;
//! - rapport de comparaison exploitable par la CLI.
//!
//! ## Dépendances
//!
//! `pmg-io`, `pmg-core`, `pmg-blueprint`, `pmg-models`.
//!
//! ## Statut
//!
//! Sprint 0 : squelette documenté, aucune API publique. Implémentation prévue
//! aux sprints 11 et 15 (lots L11, L15).
//!
//! # Exemple
//!
//! ```
//! // API métier à venir au sprint 11 ; test trivial de documentation.
//! let _ = 0u64;
//! ```

// Nouveaux modules du Sprint 15
pub mod architecture_compare;
pub mod architecture_helpers;
pub mod comparison;
pub mod config_compare;
pub mod diff;
pub mod dtype_compare;
pub mod report;
pub mod score;
pub mod shape_compare;
pub mod shard_compare;
pub mod tensor_compare;

// Réexportations des nouveaux modules
pub use architecture_compare::{
    compare_architectures, ArchitectureComparisonResult, ArchitectureType,
};
pub use comparison::{ComparisonReport, ComparisonStatus};
pub use config_compare::{
    compare_configs as compare_architecture_configs,
    ConfigComparisonResult as ArchitectureConfigComparisonResult,
};
pub use diff::{Diff, DiffCollection, DiffType};
pub use dtype_compare::{compare_dtypes, DtypeComparisonResult, DtypeInfo};
pub use report::{format_compact_report, format_detailed_report, format_report};
pub use score::{calculate_global_score, ComparisonScore};
pub use shape_compare::{compare_shapes, ShapeComparisonResult, ShapeInfo};
pub use shard_compare::{compare_sharding, ShardComparisonResult, ShardConfig, ShardInfo};
pub use tensor_compare::{compare_tensors, TensorComparisonResult, TensorInfo};
