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

//! @deprecated : Utilisez `core_config` à la place.
//! Ce module sera supprimé dans une version future.
//!
//! Conformité : ADR-002, étape 1 - Split GeneratorConfig.
//! Ce module est devenu un re-export temporaire pour assurer la transition
//! sans cassure API. Les types fondamentaux ont été déplacés dans `core_config`.

// Re-exports pour transition depuis core_config
pub use crate::core_config::GenerationMode;
pub use crate::core_config::{AmplitudeStrategy, CoreConfig as GeneratorConfig, OutlierConfig};

// Ré-export de DistributionConfig et StructureConfig pour compatibilité
pub use crate::distribution_config::DistributionConfig;
pub use crate::structure_config::StructureConfig;
