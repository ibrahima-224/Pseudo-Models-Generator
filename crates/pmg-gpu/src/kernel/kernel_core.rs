//! Noyau de compilation et exécution des kernels GPU
//!
//! Ce module contient la structure `GpuKernel` qui représente un kernel CUDA/PTX
//! compilé et prêt à l'exécution. Il gère la compilation du code PTX et
//! l'exécution sur les devices GPU.

use super::kernel_args::ToGpuArg;
use super::kernel_config::KernelConfig;
use crate::device::GpuDevice;
use crate::error::GpuResult;

/// Kernel GPU
///
/// Représente un kernel CUDA/PTX compilé et prêt à l'exécution.
pub struct GpuKernel {
    /// Nom du kernel
    name: String,
    /// Code PTX source
    ptx_code: String,
    /// Module CUDA compilé (uniquement avec la feature gpu)
    #[cfg(feature = "gpu")]
    module: cust::module::Module,
}

impl GpuKernel {
    /// Crée un nouveau kernel depuis du code PTX
    ///
    /// # Arguments
    ///
    /// * `name` - Nom du kernel (doit correspondre au nom dans le PTX)
    /// * `ptx_code` - Code PTX compilé
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::PtxCompilationError` si la compilation échoue.
    ///
    /// # Exemple
    ///
    /// ```rust,no_run
    /// use pmg_gpu::GpuKernel;
    ///
    /// let ptx = r#"
    /// .version 7.0
    /// .target sm_70
    /// .address_size 64
    ///
    /// .visible .entry my_kernel(.param .u64 output) {
    ///     .reg .u64 %rd<2>;
    ///     ld.param.u64 %rd1, [output];
    ///     st.u64 [%rd1], 42;
    ///     ret;
    /// }
    /// "#;
    ///
    /// let kernel = GpuKernel::new("my_kernel", ptx).unwrap();
    /// ```
    pub fn new(name: &str, ptx_code: &str) -> GpuResult<Self> {
        #[cfg(feature = "gpu")]
        {
            // Compiler le code PTX
            let module = cust::module::Module::from_ptx(ptx_code, &[]).map_err(|e| {
                GpuError::PtxCompilationError(format!("Compilation PTX échouée: {}", e))
            })?;

            Ok(Self {
                name: name.to_string(),
                ptx_code: ptx_code.to_string(),
                module,
            })
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mode fallback: stocker le code PTX sans compiler
            log::debug!("Création kernel (mode fallback): {}", name);

            Ok(Self {
                name: name.to_string(),
                ptx_code: ptx_code.to_string(),
            })
        }
    }

    /// Exécute le kernel sur un device
    ///
    /// # Arguments
    ///
    /// * `device` - Device GPU cible
    /// * `config` - Configuration d'exécution
    /// * `args` - Arguments du kernel
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::KernelError` si l'exécution échoue.
    ///
    /// # Sécurité
    ///
    /// Cette fonction est unsafe car elle exécute du code sur le GPU.
    /// Les arguments doivent être valides et de la bonne taille.
    pub fn launch(
        &self,
        _device: &GpuDevice,
        config: KernelConfig,
        _args: &[&dyn ToGpuArg],
    ) -> GpuResult<()> {
        // Valider la configuration
        config.validate()?;

        #[cfg(feature = "gpu")]
        {
            // Obtenir la fonction du kernel
            let function = self.module.get_function(&self.name).map_err(|e| {
                GpuError::KernelError(format!("Fonction '{}' non trouvée: {}", self.name, e))
            })?;

            // Calculer la taille de la grille si nécessaire
            let grid_dim = if config.grid_size == 0 {
                // Auto-calcul basé sur la taille des données
                if let Some(first_arg) = _args.first() {
                    (first_arg.size() as u32 + config.block_size - 1) / config.block_size
                } else {
                    1
                }
            } else {
                config.grid_size
            };

            // Préparer les arguments
            let mut kernel_args: Vec<*mut std::ffi::c_void> =
                _args.iter().map(|a| a.to_gpu_ptr()).collect();

            // Lancer le kernel
            // SAFETY : Cette opération unsafe est nécessaire pour appeler la fonction GPU.
            // Les invariants suivants sont garantis :
            // 1. Les pointeurs dans kernel_args sont valides et non-null (validés lors de la création des KernelArgs).
            // 2. Les pointeurs sont correctement alignés (validation effectuée dans les KernelArgs).
            // 3. La durée de vie des données pointées est suffisante pour toute l'exécution du kernel.
            // 4. La fonction launch est thread-safe et ne modifie pas l'état interne du module cust.
            // 5. La configuration de lancement (grid_dim, block_dim, shared_memory) est valide et testée.
            unsafe {
                let launch_config = cust::LaunchConfiguration {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (config.block_size, 1, 1),
                    shared_mem_bytes: config.shared_memory,
                };

                function
                    .launch(launch_config, &mut kernel_args)
                    .map_err(|e| {
                        GpuError::KernelError(format!("Lancement kernel échoué: {}", e))
                    })?;
            }

            log::debug!(
                "Kernel '{}' lancé: grid={}, block={}",
                self.name,
                grid_dim,
                config.block_size
            );

            Ok(())
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mode fallback: simuler l'exécution
            log::info!(
                "Simulation exécution kernel '{}' (mode fallback)",
                self.name
            );

            // Simuler un délai pour les tests de performance
            std::thread::sleep(std::time::Duration::from_micros(10));

            Ok(())
        }
    }

    /// Retourne le nom du kernel
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Retourne le code PTX source
    pub fn ptx_code(&self) -> &str {
        &self.ptx_code
    }

    /// Vérifie si le kernel est compilé (feature gpu activée)
    pub fn is_compiled(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            true
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

impl std::fmt::Debug for GpuKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuKernel")
            .field("name", &self.name)
            .field("ptx_size", &self.ptx_code.len())
            .field("compiled", &self.is_compiled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = GpuKernel::new("test_kernel", "test_ptx");
        assert!(kernel.is_ok());

        let kernel = kernel.unwrap();
        assert_eq!(kernel.name(), "test_kernel");
        assert!(!kernel.is_compiled()); // Sans feature gpu
    }

    #[test]
    fn test_kernel_launch_simulation() {
        let device = crate::device::GpuDevice::new(0).unwrap();
        let kernel = GpuKernel::new("test_kernel", "test_ptx").unwrap();
        let config = KernelConfig::default();

        // Ce test vérifie que la simulation ne panic pas
        let result = kernel.launch(&device, config, &[]);
        assert!(result.is_ok());
    }
}
