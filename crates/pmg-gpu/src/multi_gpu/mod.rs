//! Gestion multi-GPU
//!
//! Ce module fournit des fonctionnalités pour gérer plusieurs GPU
//! et distribuer les charges de travail de manière optimisée.
//! Il supporte les stratégies de distribution round-robin, least-used,
//! et personalisées, avec un suivi détaillé des statistiques d'utilisation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::device::{DeviceInfo, GpuDevice};
use crate::error::{GpuError, GpuResult};

/// Sous-module pour les tests multi-GPU
#[cfg(test)]
mod multi_gpu_tests;
/// Sous-module pour la distribution de travail
pub mod work_distributor;

// Réexportations pour maintenir la compatibilité API
pub use work_distributor::{create_multi_gpu_system, MultiGpuManager, WorkDistributor};

/// Stratégie de sélection de device GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStrategy {
    /// Distribution round-robin (séquentielle)
    RoundRobin,
    /// Sélection du device le moins utilisé
    LeastUsed,
    /// Sélection du device avec le plus de mémoire libre
    MostMemory,
}

/// Statistiques d'utilisation d'un device GPU
#[derive(Debug, Clone, Default)]
pub struct DeviceStats {
    /// Nombre de tâches assignées à ce device
    pub task_count: u64,
    /// Temps total d'exécution en millisecondes
    pub total_time_ms: u64,
    /// Nombre d'erreurs rencontrées
    pub error_count: u64,
    /// Mémoire totale allouée en octets
    pub allocated_memory: u64,
}

/// Pool de devices GPU
///
/// Gère un ensemble de devices GPU et fournit des méthodes
/// pour les sélectionner selon différentes stratégies.
#[derive(Debug)]
pub struct MultiGpuPool {
    /// Devices disponibles (protégés par Arc<Mutex> pour la thread-safety)
    devices: Vec<Arc<Mutex<GpuDevice>>>,
    /// Index actuel pour la sélection round-robin
    current_index: Mutex<usize>,
    /// Stratégie de sélection par défaut
    default_strategy: Mutex<DeviceStrategy>,
    /// Statistiques d'utilisation par device
    stats: Mutex<HashMap<usize, DeviceStats>>,
    /// Cache des informations device
    device_info_cache: Vec<DeviceInfo>,
}

impl MultiGpuPool {
    /// Crée un nouveau pool multi-GPU
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::GpuNotAvailable` si aucun device n'est disponible.
    ///
    /// # Exemple
    ///
    /// ```rust,no_run
    /// use pmg_gpu::multi_gpu::MultiGpuPool;
    ///
    /// let pool = MultiGpuPool::new().expect("Échec création pool multi-GPU");
    /// println!("Nombre de devices: {}", pool.device_count());
    /// ```
    pub fn new() -> GpuResult<Self> {
        let device_count = Self::detect_device_count()?;

        if device_count == 0 {
            return Err(GpuError::GpuNotAvailable);
        }

        let mut devices = Vec::with_capacity(device_count);
        let mut device_info_cache = Vec::with_capacity(device_count);

        for i in 0..device_count {
            let device = GpuDevice::new(i)?;
            device_info_cache.push(device.info().clone());
            devices.push(Arc::new(Mutex::new(device)));
        }

        // Initialiser les statistiques pour chaque device
        let mut stats = HashMap::new();
        for i in 0..device_count {
            stats.insert(i, DeviceStats::default());
        }

        Ok(Self {
            devices,
            current_index: Mutex::new(0),
            default_strategy: Mutex::new(DeviceStrategy::RoundRobin),
            stats: Mutex::new(stats),
            device_info_cache,
        })
    }

    /// Crée un pool avec une stratégie par défaut spécifique
    pub fn with_strategy(strategy: DeviceStrategy) -> GpuResult<Self> {
        let pool = Self::new()?;
        *pool.default_strategy.lock().unwrap() = strategy;
        Ok(pool)
    }

    /// Détecte le nombre de devices GPU disponibles
    fn detect_device_count() -> GpuResult<usize> {
        #[cfg(feature = "cuda")]
        {
            cust::driver::CudaDevice::count()
                .map(|count| count as usize)
                .map_err(|e| GpuError::CudaError(format!("Détection devices échouée: {}", e)))
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Mode fallback: simuler un device pour les tests
            log::info!("Mode fallback: simulation d'un device GPU");
            Ok(1)
        }
    }

    /// Retourne le nombre de devices dans le pool
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Retourne les informations de tous les devices sous forme de slice immuable.
    /// Cette méthode évite la copie des données en retournant une référence directe au cache.
    /// Le cache est initialisé à la création du pool et jamais modifié, garantissant la cohérence.
    pub fn devices_info(&self) -> &[DeviceInfo] {
        &self.device_info_cache
    }

    /// Obtient un device selon la stratégie spécifiée
    ///
    /// # Arguments
    ///
    /// * `strategy` - Stratégie de sélection à utiliser
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::MultiGpuError` si aucun device n'est disponible.
    pub fn get_device(&self, strategy: DeviceStrategy) -> GpuResult<Arc<Mutex<GpuDevice>>> {
        if self.devices.is_empty() {
            return Err(GpuError::MultiGpuError(
                "Aucun device disponible".to_string(),
            ));
        }

        match strategy {
            DeviceStrategy::RoundRobin => self.next_device_round_robin(),
            DeviceStrategy::LeastUsed => self.least_used_device(),
            DeviceStrategy::MostMemory => self.most_memory_device(),
        }
    }

    /// Obtient le prochain device en round-robin
    fn next_device_round_robin(&self) -> GpuResult<Arc<Mutex<GpuDevice>>> {
        let mut index = self.current_index.lock().unwrap();
        let device = self.devices[*index].clone();

        // Mettre à jour les statistiques
        if let Ok(mut stats) = self.stats.lock() {
            if let Some(device_stats) = stats.get_mut(&*index) {
                device_stats.task_count += 1;
            }
        }

        // Passer au device suivant
        *index = (*index + 1) % self.devices.len();

        Ok(device)
    }

    /// Obtient le device le moins utilisé
    fn least_used_device(&self) -> GpuResult<Arc<Mutex<GpuDevice>>> {
        let stats = self.stats.lock().unwrap();

        // Trouver le device avec le moins de tâches
        let min_usage = stats.values().map(|s| s.task_count).min().unwrap_or(0);

        for (index, device_stats) in stats.iter() {
            // Fusion des conditions if imbriquées en un seul if avec &&
            if device_stats.task_count == min_usage && *index < self.devices.len() {
                return Ok(self.devices[*index].clone());
            }
        }

        // Fallback: premier device
        Ok(self.devices[0].clone())
    }

    /// Obtient le device avec le plus de mémoire libre
    fn most_memory_device(&self) -> GpuResult<Arc<Mutex<GpuDevice>>> {
        let mut max_memory = 0;
        let mut best_device = 0;

        for (i, info) in self.device_info_cache.iter().enumerate() {
            if info.free_memory > max_memory {
                max_memory = info.free_memory;
                best_device = i;
            }
        }

        Ok(self.devices[best_device].clone())
    }

    /// Obtient un device avec la stratégie par défaut
    pub fn next_device(&self) -> GpuResult<Arc<Mutex<GpuDevice>>> {
        let strategy = *self.default_strategy.lock().unwrap();
        self.get_device(strategy)
    }

    /// Met à jour les statistiques d'un device
    pub fn update_stats(
        &self,
        device_id: usize,
        time_ms: u64,
        memory_allocated: u64,
        had_error: bool,
    ) {
        if let Ok(mut stats) = self.stats.lock() {
            if let Some(device_stats) = stats.get_mut(&device_id) {
                device_stats.total_time_ms += time_ms;
                device_stats.allocated_memory += memory_allocated;
                if had_error {
                    device_stats.error_count += 1;
                }
            }
        }
    }

    /// Retourne les statistiques d'un device spécifique
    pub fn device_stats(&self, device_id: usize) -> Option<DeviceStats> {
        self.stats.lock().unwrap().get(&device_id).cloned()
    }

    /// Retourne les statistiques de tous les devices
    pub fn all_stats(&self) -> HashMap<usize, DeviceStats> {
        self.stats.lock().unwrap().clone()
    }

    /// Réinitialise les statistiques de tous les devices
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            for device_stats in stats.values_mut() {
                *device_stats = DeviceStats::default();
            }
        }
    }

    /// Change la stratégie par défaut
    pub fn set_default_strategy(&self, strategy: DeviceStrategy) {
        *self.default_strategy.lock().unwrap() = strategy;
    }

    /// Retourne la stratégie par défaut actuelle
    pub fn default_strategy(&self) -> DeviceStrategy {
        *self.default_strategy.lock().unwrap()
    }
}

impl Default for MultiGpuPool {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            log::warn!("Échec création pool multi-GPU, pool vide créé");
            Self {
                devices: Vec::new(),
                current_index: Mutex::new(0),
                default_strategy: Mutex::new(DeviceStrategy::RoundRobin),
                stats: Mutex::new(HashMap::new()),
                device_info_cache: Vec::new(),
            }
        })
    }
}
