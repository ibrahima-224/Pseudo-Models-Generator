//! Configuration d'exécution des kernels GPU
//!
//! Ce module définit la structure `KernelConfig` qui contrôle les paramètres
//! d'exécution des kernels CUDA/PTX, comme la taille des blocs, la grille
//! et la mémoire partagée.

use crate::error::{GpuError, GpuResult};

/// Configuration d'un kernel
///
/// Définit les paramètres d'exécution d'un kernel GPU.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Taille du bloc de threads
    pub block_size: u32,
    /// Taille de la grille (0 = auto-calculé)
    pub grid_size: u32,
    /// Mémoire partagée allouée en octets
    pub shared_memory: usize,
    /// Identifiant du stream (None = stream par défaut)
    pub stream: Option<u32>,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 0,
            shared_memory: 0,
            stream: None,
        }
    }
}

impl KernelConfig {
    /// Crée une configuration pour un kernel avec auto-calcul de la grille
    pub fn auto_grid(block_size: u32) -> Self {
        Self {
            block_size,
            grid_size: 0,
            shared_memory: 0,
            stream: None,
        }
    }

    /// Crée une configuration avec mémoire partagée
    pub fn with_shared_memory(block_size: u32, shared_memory: usize) -> Self {
        Self {
            block_size,
            grid_size: 0,
            shared_memory,
            stream: None,
        }
    }

    /// Valide la configuration
    pub fn validate(&self) -> GpuResult<()> {
        if self.block_size == 0 || self.block_size > 1024 {
            return Err(GpuError::ValidationError(format!(
                "Taille de bloc invalide: {}",
                self.block_size
            )));
        }

        if self.shared_memory > 48 * 1024 {
            log::warn!("Mémoire partagée importante: {} octets", self.shared_memory);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.block_size, 256);
        assert_eq!(config.grid_size, 0);
    }

    #[test]
    fn test_kernel_config_validation() {
        let mut config = KernelConfig::default();
        assert!(config.validate().is_ok());

        config.block_size = 0;
        assert!(config.validate().is_err());

        config.block_size = 2048;
        assert!(config.validate().is_err());
    }
}
