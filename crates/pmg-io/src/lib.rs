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

//! Crate `pmg-io` — entrées-sorties : Safetensors, JSON, HTTP Range, filesystem.
//!
//! Toutes les lectures de modèles respectent le principe **Zero-Payload** :
//! les métadonnées (headers, index, configs) sont lues, jamais le contenu des
//! poids.
//!
//! ## Responsabilité
//!
//! - Writer Safetensors **streaming** (réserve d'en-tête, offsets contigus,
//!   `END−BEGIN = generated_bytes`), sharding multi-fichiers, index
//!   `model.safetensors.index.json` ;
//! - Reader **header-only** : `u64 LE` + header JSON (max 8 MiB), calcul des
//!   tailles/offsets, **jamais** de lecture du payload ;
//! - parsing des fichiers de configuration (config.json, generation_config.json,
//!   tokenizer.json, tokenizer_config.json, special_tokens_map.json,
//!   chat_template.jinja, model.safetensors.index.json) ;
//! - HTTP Range `METADATA_ONLY` : détection `206 Partial Content` vs `200 OK`
//!   (`ERR_RANGE_UNSUPPORTED`, jamais de téléchargement complet) ;
//! - Création de la structure de dossier de sortie pour les pseudo-modèles
//!   générés (atomicité, copie des configs, manifeste, artefacts d'analyse).
//!
//! ## Dépendances
//!
//! `pmg-core`. Interdit : statistiques, génération, CLI.
//!
//! ## Statut
//!
//! Sprint 10 : implémentation des writers de config et métadonnées.
//!
//! # Exemple
//!
//! ```
//! use pmg_io::config_writer::write_config;
//! use pmg_core::generator_config::GeneratorConfig;
//!
//! let config = GeneratorConfig::default();
//! let json = write_config(&config).unwrap();
//! assert!(json.contains("\"seed\""));
//! ```

pub mod config_writer;
pub mod metadata_writer;
pub mod output_structure;
pub mod pool;
pub mod safetensors;
pub mod statistical_profile;

// Module HTTP Range conditionnel à la feature http-range
#[cfg(feature = "http-range")]
pub mod http_range;

// Réexportations pour faciliter l'usage
pub use output_structure::{
    atomic_write, copy_config_files, create_output_structure, create_pmg_subdirectory,
    write_pmg_metadata, OutputConfig, SourceModel,
};

// Réexportation du module statistical_profile
pub use statistical_profile::{
    load_from_file as load_statistical_profile, StatisticalProfileError,
};

// Réexportations conditionnelles pour HTTP Range
#[cfg(feature = "http-range")]
pub use http_range::{
    cache_metadata, check_range_support, fetch_metadata_only, invalidate_cache,
    load_cached_metadata, parse_header, CachedMetadata, HttpRangeConfig, HttpRangeError,
};

#[cfg(test)]
mod tests {
    #[test]
    fn skeleton_compiles() {
        // Test de compilation du module.
        let _ = 0u64;
    }
}
