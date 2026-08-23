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

//! Sous-module contenant les types de configuration et les métadonnées en cache.

use std::path::PathBuf;
use std::time::Duration;

/// Configuration du client HTTP Range.
#[derive(Debug, Clone)]
pub struct HttpRangeConfig {
    /// Timeout pour les requêtes HTTP (défaut: 30 secondes).
    pub timeout: Duration,
    /// Répertoire de cache (défaut: `~/.cache/pmg/metadata/`).
    pub cache_dir: PathBuf,
    /// Taille maximale du header (défaut: 8 MiB).
    pub max_header_size: usize,
    /// Nombre maximal de tentatives en cas d'échec (défaut: 3).
    pub max_retries: u32,
    /// User-Agent pour les requêtes.
    pub user_agent: String,
}

impl Default for HttpRangeConfig {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("pmg")
            .join("metadata");

        Self {
            timeout: Duration::from_secs(30),
            cache_dir,
            max_header_size: 8 * 1024 * 1024, // 8 MiB
            max_retries: 3,
            user_agent: format!("pmg-io/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

/// Métadonnées en cache d'un fichier Safetensors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedMetadata {
    /// URL source du fichier.
    pub url: String,
    /// Hash SHA-256 de l'URL (pour nommage du cache).
    pub url_hash: String,
    /// Header JSON du fichier Safetensors.
    pub header_json: String,
    /// Taille du header en octets.
    pub header_size: u64,
    /// Timestamp de la mise en cache (Unix timestamp).
    pub cached_at: u64,
    /// Version du format de cache.
    pub cache_version: u32,
}

impl CachedMetadata {
    /// Crée une nouvelle instance de métadonnées en cache.
    pub fn new(url: &str, header_json: String, header_size: u64) -> Self {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(url.as_bytes());
        let url_hash = format!("{:x}", hasher.finalize());

        let cached_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            url: url.to_string(),
            url_hash,
            header_json,
            header_size,
            cached_at,
            cache_version: 1,
        }
    }

    /// Retourne le chemin du fichier de cache pour cette URL.
    pub fn cache_path(&self, cache_dir: &std::path::Path) -> PathBuf {
        cache_dir.join(format!("{}.json", self.url_hash))
    }
}
