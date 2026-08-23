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

//! Sous-module contenant les types de rapport d'inspection structurés.
//!
//! Ce module fournit la structure [`StructuredReport`] pour les rapports
//! d'inspection structurés, ainsi que des méthodes de génération en format
//! texte et JSON.

// Sous-modules
mod report_formatters;
mod report_impl;
mod types;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use report_formatters::{format_bytes, format_number};
pub use types::{
    ArchitectureSummaryJson, ConfigInspectionJson, MoEConfigJson, PhysicalStatsJson,
    SafetensorsHeaderJson, ShardIndexJson, StructuralStatsJson, StructuredReport, TensorInfoJson,
};

// Tests unitaires
#[cfg(test)]
mod tests;
