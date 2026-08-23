//! Erreurs de compression

use thiserror::Error;

/// Erreur de compression
#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("Algorithme non supporté: {0}")]
    UnsupportedAlgorithm(String),

    #[error("Erreur de compression: {0}")]
    CompressionFailed(String),

    #[error("Erreur de décompression: {0}")]
    DecompressionFailed(String),

    #[error("Erreur d'E/S: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Données corrompues")]
    CorruptedData,

    #[error("Taille trop grande pour la compression")]
    DataTooLarge,

    #[error("Configuration invalide")]
    InvalidConfig,
}

/// Résultat de compression
pub type CompressionResult<T> = Result<T, CompressionError>;
