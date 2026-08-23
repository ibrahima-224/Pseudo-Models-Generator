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

//! Sous-module contenant les erreurs spécifiques au module HTTP Range.

use pmg_core::error::CoreError;

/// Erreurs spécifiques au module HTTP Range.
#[derive(Debug, thiserror::Error)]
pub enum HttpRangeError {
    /// Erreur réseau (connexion, timeout, etc.).
    #[error("Erreur réseau: {0}")]
    Network(String),

    /// Le serveur ne supporte pas les requêtes Range.
    #[error("Le serveur ne supporte pas les requêtes Range (réponse {status})")]
    RangeUnsupported { status: u16 },

    /// Réponse inattendue du serveur.
    #[error("Réponse inattendue: {message}")]
    UnexpectedResponse { message: String },

    /// Header Safetensors invalide ou corrompu.
    #[error("Header Safetensors invalide: {0}")]
    InvalidHeader(String),

    /// Erreur de cache (lecture/écriture).
    #[error("Erreur de cache: {0}")]
    CacheError(String),

    /// Taille du header dépasse la limite autorisée.
    #[error("Taille du header ({size} octets) dépasse la limite ({limit} octets)")]
    HeaderTooLarge { size: u64, limit: usize },

    /// Erreur de sérialisation/désérialisation.
    #[error("Erreur de sérialisation: {0}")]
    SerializationError(String),
}

impl From<HttpRangeError> for CoreError {
    fn from(err: HttpRangeError) -> Self {
        match err {
            HttpRangeError::Network(msg) => CoreError::Internal(format!("Réseau: {msg}")),
            HttpRangeError::RangeUnsupported { status } => {
                CoreError::Internal(format!("Range non supporté (status {status})"))
            },
            HttpRangeError::UnexpectedResponse { message } => {
                CoreError::Internal(format!("Réponse inattendue: {message}"))
            },
            HttpRangeError::InvalidHeader(msg) => CoreError::InvalidShape(format!("Header: {msg}")),
            HttpRangeError::CacheError(msg) => CoreError::Internal(format!("Cache: {msg}")),
            HttpRangeError::HeaderTooLarge { size, limit } => {
                CoreError::Overflow(format!("Header trop grand: {size} > {limit}"))
            },
            HttpRangeError::SerializationError(msg) => {
                CoreError::Internal(format!("Sérialisation: {msg}"))
            },
        }
    }
}
