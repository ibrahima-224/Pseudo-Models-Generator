//! Erreurs GPU pour PMG
//!
//! Ce module définit les types d'erreurs spécifiques au GPU
//! pour une gestion robuste et typée des erreurs.

use std::fmt;

/// Erreur GPU
///
/// Cette énumération couvre tous les types d'erreurs pouvant survenir
/// lors de l'utilisation des fonctionnalités GPU.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU non disponible ou non initialisé
    GpuNotAvailable,

    /// Erreur CUDA spécifique
    CudaError(String),

    /// Erreur d'allocation mémoire GPU
    AllocationError(String),

    /// Erreur d'exécution de kernel
    KernelError(String),

    /// Erreur de compilation PTX
    PtxCompilationError(String),

    /// Erreur liée au multi-GPU
    MultiGpuError(String),

    /// Timeout lors d'une opération GPU
    GpuTimeout,

    /// Erreur de transfert mémoire host-device
    TransferError(String),

    /// Erreur de validation des paramètres
    ValidationError(String),

    /// Erreur interne inattendue
    InternalError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::GpuNotAvailable => write!(f, "GPU non disponible"),
            GpuError::CudaError(msg) => write!(f, "Erreur CUDA: {}", msg),
            GpuError::AllocationError(msg) => write!(f, "Erreur d'allocation mémoire: {}", msg),
            GpuError::KernelError(msg) => write!(f, "Erreur de kernel: {}", msg),
            GpuError::PtxCompilationError(msg) => write!(f, "Erreur de compilation PTX: {}", msg),
            GpuError::MultiGpuError(msg) => write!(f, "Erreur multi-GPU: {}", msg),
            GpuError::GpuTimeout => write!(f, "Timeout GPU"),
            GpuError::TransferError(msg) => write!(f, "Erreur de transfert mémoire: {}", msg),
            GpuError::ValidationError(msg) => write!(f, "Erreur de validation: {}", msg),
            GpuError::InternalError(msg) => write!(f, "Erreur interne: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Pas de source sous-jacente pour l'instant
        None
    }
}

/// Résultat GPU
///
/// Type alias pour `Result<T, GpuError>`.
pub type GpuResult<T> = Result<T, GpuError>;

/// Implémentation de From pour les conversions automatiques
impl From<String> for GpuError {
    fn from(err: String) -> Self {
        GpuError::InternalError(err)
    }
}

impl From<&str> for GpuError {
    fn from(err: &str) -> Self {
        GpuError::InternalError(err.to_string())
    }
}

impl From<std::io::Error> for GpuError {
    fn from(err: std::io::Error) -> Self {
        GpuError::InternalError(format!("Erreur I/O: {}", err))
    }
}

/// Implémentation de From pour cust::CudaError si la feature gpu est activée
#[cfg(feature = "gpu")]
impl From<cust::CudaError> for GpuError {
    fn from(err: cust::CudaError) -> Self {
        GpuError::CudaError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_error_display() {
        let error = GpuError::GpuNotAvailable;
        assert_eq!(error.to_string(), "GPU non disponible");

        let error = GpuError::CudaError("test".to_string());
        assert_eq!(error.to_string(), "Erreur CUDA: test");
    }

    #[test]
    fn test_gpu_error_conversion() {
        let error: GpuError = "erreur test".into();
        match error {
            GpuError::InternalError(msg) => assert_eq!(msg, "erreur test"),
            _ => panic!("Type d'erreur incorrect"),
        }
    }

    #[test]
    fn test_gpu_error_clone() {
        let error = GpuError::AllocationError("test".to_string());
        let cloned = error.clone();
        assert_eq!(error.to_string(), cloned.to_string());
    }
}
