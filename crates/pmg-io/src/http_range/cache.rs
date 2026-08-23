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

//! Sous-module contenant les opérations de cache pour les métadonnées HTTP Range.

use pmg_core::error::CoreResult;

use super::config::{CachedMetadata, HttpRangeConfig};
use super::error::HttpRangeError;

/// Met en cache les métadonnées d'un fichier Safetensors.
///
/// # Paramètres
/// - `metadata` : métadonnées à mettre en cache
/// - `config` : configuration du client HTTP Range
///
/// # Erreurs
/// Retourne une erreur si l'écriture du cache échoue.
pub fn cache_metadata(metadata: &CachedMetadata, config: &HttpRangeConfig) -> CoreResult<()> {
    // Crée le répertoire de cache s'il n'existe pas
    std::fs::create_dir_all(&config.cache_dir)
        .map_err(|e| HttpRangeError::CacheError(format!("Création répertoire: {e}")))?;

    let cache_path = metadata.cache_path(&config.cache_dir);
    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| HttpRangeError::SerializationError(format!("Sérialisation: {e}")))?;

    std::fs::write(&cache_path, json)
        .map_err(|e| HttpRangeError::CacheError(format!("Écriture cache: {e}")))?;

    Ok(())
}

/// Charge les métadonnées en cache pour une URL donnée.
///
/// # Paramètres
/// - `url` : URL du fichier Safetensors
/// - `config` : configuration du client HTTP Range
///
/// # Retourne
/// `Some(CachedMetadata)` si le cache existe, `None` sinon.
pub fn load_cached_metadata(
    url: &str,
    config: &HttpRangeConfig,
) -> CoreResult<Option<CachedMetadata>> {
    use sha2::{Digest, Sha256};

    // Calcule le hash de l'URL
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let url_hash = format!("{:x}", hasher.finalize());

    let cache_path = config.cache_dir.join(format!("{url_hash}.json"));

    if !cache_path.exists() {
        return Ok(None);
    }

    let json = std::fs::read_to_string(&cache_path)
        .map_err(|e| HttpRangeError::CacheError(format!("Lecture cache: {e}")))?;

    let metadata: CachedMetadata = serde_json::from_str(&json)
        .map_err(|e| HttpRangeError::CacheError(format!("Désérialisation cache: {e}")))?;

    // Vérifie que le cache est pour la bonne URL
    if metadata.url != url {
        return Err(HttpRangeError::CacheError("URL du cache incohérente".to_string()).into());
    }

    Ok(Some(metadata))
}

/// Supprime le cache pour une URL donnée.
///
/// # Paramètres
/// - `url` : URL du fichier Safetensors
/// - `config` : configuration du client HTTP Range
///
/// # Retourne
/// `true` si le cache a été supprimé, `false` s'il n'existait pas.
pub fn invalidate_cache(url: &str, config: &HttpRangeConfig) -> CoreResult<bool> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let url_hash = format!("{:x}", hasher.finalize());

    let cache_path = config.cache_dir.join(format!("{url_hash}.json"));

    if cache_path.exists() {
        std::fs::remove_file(&cache_path)
            .map_err(|e| HttpRangeError::CacheError(format!("Suppression cache: {e}")))?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Vérifie si le serveur supporte les requêtes Range pour une URL donnée.
///
/// # Paramètres
/// - `url` : URL à tester
/// - `config` : configuration du client HTTP Range
///
/// # Retourne
/// `true` si le serveur supporte Range, `false` sinon.
pub fn check_range_support(url: &str, config: &HttpRangeConfig) -> CoreResult<bool> {
    let client = reqwest::blocking::Client::builder()
        .timeout(config.timeout)
        .user_agent(&config.user_agent)
        .build()
        .map_err(|e| HttpRangeError::Network(format!("Impossible de créer le client: {e}")))?;

    let response = client
        .get(url)
        .header("Range", "bytes=0-0")
        .send()
        .map_err(|e| HttpRangeError::Network(format!("Échec requête: {e}")))?;

    // Le serveur supporte Range si la réponse est 206 Partial Content
    Ok(response.status() == 206)
}
