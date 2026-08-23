//! Module GPU pour PMG
//!
//! Ce module fournit l'accélération GPU pour la génération de modèles.
//! Il inclut des kernels CUDA/PTX pour les opérations intensives.
//!
//! # Architecture
//!
//! Le module est organisé en plusieurs sous-modules :
//! - `error` : Gestion des erreurs GPU
//! - `device` : Gestion des devices GPU
//! - `kernel` : Kernels CUDA/PTX
//! - `allocator` : Allocations mémoire GPU
//! - `multi_gpu` : Support multi-GPU
//!
//! # Features
//!
//! - `gpu` : Active le support GPU (nécessite CUDA)
//! - `cpu-fallback` : Active le fallback CPU (par défaut)
//!
//! # Exemple d'utilisation
//!
//! ```rust,no_run
//! use pmg_gpu::{GpuDevice, MultiGpuManager};
//!
//! // Vérifier la disponibilité du GPU
//! if pmg_gpu::is_gpu_available() {
//!     // Créer un gestionnaire multi-GPU
//!     let manager = MultiGpuManager::new().unwrap();
//!     println!("GPU disponibles: {}", manager.device_count());
//! }
//! ```

pub mod acceleration;
pub mod allocator;
pub mod device;
pub mod error;
pub mod kernel;
pub mod kernels;
pub mod metrics;
pub mod multi_gpu;
pub mod performance_tests;
pub mod security_tests;

pub use acceleration::GpuAccelerated;
pub use allocator::GpuAllocator;
pub use device::{DeviceInfo, GpuDevice};
pub use error::{GpuError, GpuResult};
pub use kernel::{GpuKernel, KernelConfig, ToGpuArg};
pub use kernels::NormalGenerationAccelerated;
pub use metrics::{OperationTimer, PerformanceMetrics, PerformanceReport};
pub use multi_gpu::MultiGpuManager;

/// Version du module GPU
pub const GPU_MODULE_VERSION: &str = "0.1.0";

/// Vérifie la disponibilité du GPU
///
/// # Retour
///
/// `true` si un GPU est disponible et peut être initialisé, `false` sinon.
///
/// # Exemple
///
/// ```rust
/// if pmg_gpu::is_gpu_available() {
///     println!("GPU disponible");
/// } else {
///     println!("Pas de GPU, utilisation du CPU");
/// }
/// ```
pub fn is_gpu_available() -> bool {
    // Vérifier si CUDA est disponible via le crate cust
    #[cfg(feature = "cuda")]
    {
        // Tenter d'initialiser le device GPU principal (index 0)
        // Cette opération est rapide (< 100ms) et ne nécessite pas de nettoyage explicite
        match cust::driver::CudaDevice::get(0) {
            Ok(_) => true,   // Device GPU trouvé et initialisé avec succès
            Err(_) => false, // Aucun device GPU disponible ou erreur d'initialisation
        }
    }

    // Fallback : retourner false si le feature cuda n'est pas activé
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Nombre maximum de GPU supportés
pub const MAX_GPU_DEVICES: usize = 8;

/// Taille minimale de bloc pour les kernels
pub const MIN_BLOCK_SIZE: u32 = 32;

/// Taille maximale de bloc pour les kernels
pub const MAX_BLOCK_SIZE: u32 = 1024;

/// Taille par défaut du bloc pour les kernels
pub const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Constantes pour les kernels
pub mod constants {
    /// Nombre de threads par warp (architecture NVIDIA)
    pub const WARP_SIZE: u32 = 32;

    /// Taille de la mémoire partagée par défaut (en octets)
    pub const DEFAULT_SHARED_MEMORY: usize = 48 * 1024; // 48 KB

    /// Nombre maximum de blocs par grille (1D)
    pub const MAX_GRID_DIM_X: u32 = 2_147_483_647; // 2^31 - 1

    /// Nombre maximum de threads par bloc
    pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        // Ce test vérifie que la fonction ne panic pas
        let _available = is_gpu_available();
    }

    #[test]
    fn test_version() {
        assert_eq!(GPU_MODULE_VERSION, "0.1.0");
    }

    #[test]
    fn test_constants() {
        assert_eq!(constants::WARP_SIZE, 32);
        assert!(constants::MAX_THREADS_PER_BLOCK >= 512);
        assert!(constants::MAX_GRID_DIM_X > 0);
    }
}
