//! Registre et lifecycle des kernels précompilés
//!
//! Ce module fournit un registre centralisé pour gérer les kernels GPU précompilés.
//! Il permet d'enregistrer, de récupérer et de lister les kernels disponibles.

use super::kernel_core::GpuKernel;
use super::kernel_ptx::{
    BF16_CONVERSION_KERNEL, MATRIX_MULTIPLICATION_KERNEL, MIXTURE_DISTRIBUTION_KERNEL,
    NORMAL_GENERATION_KERNEL,
};
use crate::error::GpuResult;

/// Registre de kernels précompilés
///
/// Fournit un accès rapide aux kernels les plus utilisés.
pub struct KernelRegistry {
    kernels: std::collections::HashMap<String, GpuKernel>,
}

impl KernelRegistry {
    /// Crée un nouveau registre avec les kernels par défaut
    pub fn new() -> GpuResult<Self> {
        let mut kernels = std::collections::HashMap::new();

        // Enregistrer les kernels prédéfinis
        kernels.insert(
            "normal_generation".to_string(),
            GpuKernel::new("normal_generation_kernel", NORMAL_GENERATION_KERNEL)?,
        );

        kernels.insert(
            "bf16_conversion".to_string(),
            GpuKernel::new("f32_to_bf16_kernel", BF16_CONVERSION_KERNEL)?,
        );

        kernels.insert(
            "mixture_distribution".to_string(),
            GpuKernel::new("mixture_distribution_kernel", MIXTURE_DISTRIBUTION_KERNEL)?,
        );

        kernels.insert(
            "matrix_multiplication".to_string(),
            GpuKernel::new("matrix_multiplication_kernel", MATRIX_MULTIPLICATION_KERNEL)?,
        );

        Ok(Self { kernels })
    }

    /// Retourne un kernel par son nom
    pub fn get(&self, name: &str) -> Option<&GpuKernel> {
        self.kernels.get(name)
    }

    /// Enregistre un kernel personnalisé
    pub fn register(&mut self, name: String, kernel: GpuKernel) {
        self.kernels.insert(name, kernel);
    }

    /// Retourne la liste des kernels disponibles
    pub fn available_kernels(&self) -> Vec<&str> {
        self.kernels.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            log::warn!("Échec création registre kernels par défaut");
            Self {
                kernels: std::collections::HashMap::new(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_registry() {
        let registry = KernelRegistry::new();
        assert!(registry.is_ok());

        let registry = registry.unwrap();
        assert!(!registry.available_kernels().is_empty());
        assert!(registry.get("normal_generation").is_some());
    }
}
