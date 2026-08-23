//! Support GPU pour PMG
//!
//! Ce module fournit l'intégration GPU pour le générateur de modèles.
//! Il permet d'utiliser l'accélération GPU lorsque disponible,
//! avec fallback automatique sur CPU.

use crate::generator_config::GeneratorConfig;

/// Configuration GPU
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Activer l'accélération GPU
    pub enabled: bool,
    /// Nombre de GPU à utiliser
    pub gpu_count: Option<usize>,
    /// Taille des blocs GPU
    pub block_size: u32,
    /// Mémoire partagée par bloc (en octets)
    pub shared_memory: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            gpu_count: None,
            block_size: 256,
            shared_memory: 0,
        }
    }
}

/// Statut GPU
#[derive(Debug, Clone)]
pub enum GpuStatus {
    /// GPU disponible et actif
    Active {
        /// Nombre de GPU actifs
        device_count: usize,
        /// Nom du premier device
        device_name: String,
    },
    /// GPU disponible mais désactivé
    Available {
        /// Nombre de GPU disponibles
        device_count: usize,
    },
    /// GPU non disponible (fallback CPU)
    Fallback {
        /// Raison du fallback
        reason: String,
    },
}

/// Gestionnaire de support GPU
///
/// Fournit une interface unifiée pour l'accélération GPU
/// avec fallback automatique sur CPU.
pub struct GpuSupportManager {
    /// Configuration GPU
    config: GpuConfig,
    /// Statut actuel
    status: GpuStatus,
}

impl GpuSupportManager {
    /// Crée un nouveau gestionnaire de support GPU
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration GPU
    pub fn new(config: GpuConfig) -> Self {
        let status = if config.enabled {
            Self::detect_gpu_status()
        } else {
            GpuStatus::Fallback {
                reason: "GPU désactivé dans la configuration".to_string(),
            }
        };

        Self { config, status }
    }

    /// Détecte le statut GPU
    fn detect_gpu_status() -> GpuStatus {
        #[cfg(feature = "gpu-acceleration")]
        {
            // Tenter d'initialiser le GPU
            match pmg_gpu::is_gpu_available() {
                true => {
                    // Essayer de créer un gestionnaire multi-GPU
                    match pmg_gpu::MultiGpuManager::new() {
                        Ok(manager) => {
                            let device_count = manager.device_count();
                            let device_name = manager
                                .devices_info()
                                .first()
                                .map(|info| info.name.clone())
                                .unwrap_or_else(|| "GPU inconnu".to_string());

                            GpuStatus::Active {
                                device_count,
                                device_name,
                            }
                        },
                        Err(e) => {
                            log::warn!("Échec initialisation multi-GPU: {}", e);
                            GpuStatus::Fallback {
                                reason: format!("Erreur initialisation GPU: {}", e),
                            }
                        },
                    }
                },
                false => GpuStatus::Fallback {
                    reason: "GPU non disponible".to_string(),
                },
            }
        }

        #[cfg(not(feature = "gpu-acceleration"))]
        {
            GpuStatus::Fallback {
                reason: "Support GPU non compilé".to_string(),
            }
        }
    }

    /// Retourne la configuration GPU
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Retourne le statut GPU
    pub fn status(&self) -> &GpuStatus {
        &self.status
    }

    /// Vérifie si le GPU est actif
    pub fn is_gpu_active(&self) -> bool {
        matches!(self.status, GpuStatus::Active { .. })
    }

    /// Retourne le nombre de GPU actifs
    pub fn active_device_count(&self) -> usize {
        match &self.status {
            GpuStatus::Active { device_count, .. } => *device_count,
            GpuStatus::Available { device_count } => *device_count,
            GpuStatus::Fallback { .. } => 0,
        }
    }

    /// Crée une configuration de génération avec support GPU
    pub fn create_generation_config(&self, base_config: GeneratorConfig) -> GeneratorConfig {
        // Pour l'instant, retourner la configuration de base
        // L'intégration complète sera ajoutée lors de l'implémentation
        // des kernels GPU dans le pipeline de génération
        base_config
    }
}

impl std::fmt::Debug for GpuSupportManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSupportManager")
            .field("config", &self.config)
            .field("status", &self.status)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_gpu_support_manager_creation() {
        let config = GpuConfig::default();
        let manager = GpuSupportManager::new(config);

        assert!(!manager.is_gpu_active());
        assert_eq!(manager.active_device_count(), 0);
    }

    #[test]
    fn test_gpu_status_detection() {
        let config = GpuConfig {
            enabled: true,
            ..Default::default()
        };

        let manager = GpuSupportManager::new(config);
        // Le statut dépend de la compilation et du système
        // Ce test vérifie simplement que la détection ne panic pas
        let _status = manager.status();
    }
}
