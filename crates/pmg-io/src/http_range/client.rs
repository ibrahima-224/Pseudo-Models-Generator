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

//! Sous-module contenant les fonctions de requête HTTP pour récupérer les métadonnées.

use pmg_core::error::CoreResult;

use super::config::{CachedMetadata, HttpRangeConfig};
use super::error::HttpRangeError;

/// Récupère uniquement les métadonnées d'un fichier Safetensors distant via HTTP Range.
///
/// # Paramètres
/// - `url` : URL du fichier Safetensors distant
/// - `config` : configuration du client HTTP Range
///
/// # Retourne
/// Les métadonnées en cache contenant le header JSON.
///
/// # Erreurs
/// Retourne une erreur si:
/// - Le serveur ne supporte pas les requêtes Range
/// - La connexion réseau échoue
/// - Le header est invalide ou trop grand
/// - La taille du header dépasse la limite
pub fn fetch_metadata_only(url: &str, config: &HttpRangeConfig) -> CoreResult<CachedMetadata> {
    // Vérifie le cache d'abord
    if let Some(cached) = super::cache::load_cached_metadata(url, config)? {
        return Ok(cached);
    }

    // Crée un client HTTP
    let client = reqwest::blocking::Client::builder()
        .timeout(config.timeout)
        .user_agent(&config.user_agent)
        .build()
        .map_err(|e| HttpRangeError::Network(format!("Impossible de créer le client: {e}")))?;

    // Étape 1: Récupère les 8 premiers octets (taille du header)
    let header_size = fetch_header_size(url, &client)?;

    // Vérifie la taille du header
    if header_size as usize > config.max_header_size {
        return Err(HttpRangeError::HeaderTooLarge {
            size: header_size,
            limit: config.max_header_size,
        }
        .into());
    }

    // Étape 2: Récupère le header complet
    let header_json = fetch_header_content(url, &client, header_size, config.max_retries)?;

    // Valide que c'est du JSON valide
    let _: serde_json::Value = serde_json::from_str(&header_json)
        .map_err(|e| HttpRangeError::InvalidHeader(format!("JSON invalide: {e}")))?;

    // Crée et met en cache les métadonnées
    let metadata = CachedMetadata::new(url, header_json, header_size);
    super::cache::cache_metadata(&metadata, config)?;

    Ok(metadata)
}

/// Récupère la taille du header (8 premiers octets) via HTTP Range.
fn fetch_header_size(url: &str, client: &reqwest::blocking::Client) -> CoreResult<u64> {
    let response = client
        .get(url)
        .header("Range", "bytes=0-7")
        .send()
        .map_err(|e| HttpRangeError::Network(format!("Échec requête: {e}")))?;

    // Vérifie que le serveur supporte Range
    if response.status() == 200 {
        return Err(HttpRangeError::RangeUnsupported { status: 200 }.into());
    }

    if response.status() != 206 {
        return Err(HttpRangeError::UnexpectedResponse {
            message: format!("Status inattendu: {}", response.status()),
        }
        .into());
    }

    let bytes = response
        .bytes()
        .map_err(|e| HttpRangeError::Network(format!("Lecture réponse: {e}")))?;

    if bytes.len() != 8 {
        return Err(HttpRangeError::UnexpectedResponse {
            message: format!("Réponse inattendue: {} octets au lieu de 8", bytes.len()),
        }
        .into());
    }

    // Convertit en u64 little-endian
    let header_size = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| {
        HttpRangeError::UnexpectedResponse {
            message: "Conversion bytes->u64 échouée".to_string(),
        }
    })?);

    Ok(header_size)
}

/// Récupère le contenu du header via HTTP Range.
///
/// Cette fonction tente de récupérer le header avec un mécanisme de retry
/// pour les erreurs réseau. Les erreurs de protocole (status != 206) sont
/// retournées immédiatement sans retry.
fn fetch_header_content(
    url: &str,
    client: &reqwest::blocking::Client,
    header_size: u64,
    max_retries: u32,
) -> CoreResult<String> {
    let start = 8;
    let end = 7 + header_size;

    let mut last_error = None;

    for attempt in 0..max_retries {
        let response = match client
            .get(url)
            .header("Range", format!("bytes={start}-{end}"))
            .send()
        {
            Ok(resp) => resp,
            Err(e) => {
                // Erreur réseau : on enregistre et on continue pour retry
                last_error = Some(HttpRangeError::Network(format!(
                    "Échec requête (tentative {}): {e}",
                    attempt + 1
                )));
                continue;
            },
        };

        if response.status() != 206 {
            return Err(HttpRangeError::UnexpectedResponse {
                message: format!("Status inattendu: {}", response.status()),
            }
            .into());
        }

        let bytes = match response.bytes() {
            Ok(b) => b,
            Err(e) => {
                last_error = Some(HttpRangeError::Network(format!(
                    "Lecture réponse (tentative {}): {e}",
                    attempt + 1
                )));
                continue;
            },
        };

        // Vérifie la taille
        if bytes.len() as u64 != header_size {
            return Err(HttpRangeError::UnexpectedResponse {
                message: format!(
                    "Taille inattendue: {} octets au lieu de {}",
                    bytes.len(),
                    header_size
                ),
            }
            .into());
        }

        // Convertit en string UTF-8
        let header_str = String::from_utf8(bytes.to_vec())
            .map_err(|e| HttpRangeError::InvalidHeader(format!("Header non UTF-8 valide: {e}")))?;

        return Ok(header_str);
    }

    // Toutes les tentatives ont échoué
    if let Some(err) = last_error {
        Err(err.into())
    } else {
        Err(HttpRangeError::Network(format!("Échec après {max_retries} tentatives")).into())
    }
}

/// Parse le header JSON d'un fichier Safetensors.
///
/// # Paramètres
/// - `header_json` : chaîne JSON du header
///
/// # Retourne
/// Le parsing du header sous forme de `serde_json::Value`.
pub fn parse_header(header_json: &str) -> CoreResult<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| HttpRangeError::InvalidHeader(format!("JSON invalide: {e}")))?;

    // Vérifie que c'est un objet JSON
    if !value.is_object() {
        return Err(
            HttpRangeError::InvalidHeader("Le header n'est pas un objet JSON".to_string()).into(),
        );
    }

    Ok(value)
}
