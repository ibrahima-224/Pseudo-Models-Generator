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

//! Crate `pmg-inspect` — commande `espec` (expertise de spécification).
//!
//! Inspection **metadata-only** d'un modèle : analyse les fichiers de
//! configuration, l'index Safetensors et la structure déclarée, **sans jamais
//! lire les poids** (principe Zero-Payload).
//!
//! ## Responsabilité
//!
//! - catégories OBSERVÉ / ESTIMÉ / GÉNÉRÉ / INCONNU et niveaux
//!   EXACT / DERIVED / ESTIMATED / SYNTHETIC / UNKNOWN ;
//! - rapports textuels (normal/verbose) et JSON optionnel (`--json`).
//!
//! ## Dépendances
//!
//! `pmg-io`, `pmg-core`, `pmg-blueprint`, `pmg-models`.
//!
//! ## Statut
//!
//! Sprint 14 : implémentation complète des modules d'inspection.
//!
//! # Exemple
//!
//! ```
//! use pmg_inspect::inspector::ModelInspector;
//!
//! // Création d'un inspecteur pour un modèle (chemin fictif)
//! // let inspector = ModelInspector::new("path/to/model");
//! // let report = inspector.inspect().unwrap();
//! // println!("{}", report);
//! ```

pub mod architecture;
pub mod config_inspector;
pub mod config_moe_parser;
pub mod display;
pub mod error;
pub mod index_inspector;
pub mod inspector;
pub mod physical_stats;
pub mod report;
pub mod safetensors_inspector;
pub mod structural_stats;

// Ré-exports pratiques
pub use error::InspectError;
pub use inspector::{InspectionLevel, InspectionReport, ModelInspector};
pub use report::StructuredReport;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect_error_display() {
        let error = InspectError::ConfigNotFound(std::path::PathBuf::from("/fake/config.json"));
        assert!(error.to_string().contains("config.json"));
    }

    #[test]
    fn test_inspection_level_default() {
        assert_eq!(InspectionLevel::default(), InspectionLevel::Normal);
    }

    #[test]
    fn test_model_inspector_creation() {
        let inspector = ModelInspector::new("/path/to/model");
        assert_eq!(
            inspector.model_path(),
            std::path::Path::new("/path/to/model")
        );
        assert_eq!(inspector.level, InspectionLevel::Normal);
    }
}
