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

//! Module HTTP Range pour récupérer les métadonnées de fichiers Safetensors distants.
//!
//! Ce module implémente le téléchargement sélectif via HTTP Range pour récupérer
//! uniquement les headers des fichiers Safetensors, conformément au principe
//! **Zero-Payload** : aucun poids réel n'est lu.
//!
//! ## Fonctionnalités
//!
//! - Requêtes `Range: bytes=0-7` puis `Range: bytes=8-(7+header_size)`
//! - Cache local dans `~/.cache/pmg/metadata/`
//! - Gestion robuste des erreurs
//! - Feature gate `http-range` (désactivée par défaut en v1.0)
//!
//! ## Exemple
//!
//! ```rust,ignore
//! use pmg_io::http_range::{HttpRangeConfig, fetch_metadata_only};
//!
//! let config = HttpRangeConfig::default();
//! let metadata = fetch_metadata_only("https://example.com/model.safetensors", &config)?;
//! println!("Header: {}", metadata.header_json);
//! ```
//!
//! # Sécurité
//!
//! - Aucun téléchargement complet de fichier
//! - Validation des réponses HTTP
//! - Cache sécurisé avec hachage SHA-256 des URLs

// Sous-modules
mod cache;
mod client;
mod config;
mod error;
#[cfg(test)]
mod tests;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use config::{CachedMetadata, HttpRangeConfig};
pub use error::HttpRangeError;

// Ré-exports des fonctions publiques depuis les sous-modules.
pub use cache::{cache_metadata, check_range_support, invalidate_cache, load_cached_metadata};
pub use client::{fetch_metadata_only, parse_header};
