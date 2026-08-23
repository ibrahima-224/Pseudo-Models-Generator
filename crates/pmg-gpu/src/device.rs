//! Gestion des devices GPU
//!
//! Ce module fournit une abstraction pour les devices GPU,
//! incluant la détection, l'initialisation et les opérations de base.

use crate::error::{GpuError, GpuResult};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Information sur un device GPU
///
/// Contient les propriétés et capacités d'un device GPU.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Identifiant du device
    pub id: usize,
    /// Nom du device (ex: "NVIDIA GeForce RTX 3080")
    pub name: String,
    /// Version du compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// Mémoire totale en octets
    pub total_memory: u64,
    /// Mémoire libre en octets (approximation)
    pub free_memory: u64,
    /// Nombre maximum de threads par bloc
    pub max_threads_per_block: u32,
    /// Nombre maximum de blocs par grille (dimension X)
    pub max_blocks_per_grid: u32,
    /// Taille du warp (généralement 32)
    pub warp_size: u32,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            id: 0,
            name: "GPU virtuel".to_string(),
            compute_capability: (7, 0),
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
            free_memory: 8 * 1024 * 1024 * 1024,
            max_threads_per_block: 1024,
            max_blocks_per_grid: 2147483647,
            warp_size: 32,
        }
    }
}

/// Pointeur GPU
///
/// Représente un bloc de mémoire alloué sur le GPU.
/// La mémoire est automatiquement libérée lors de la destruction.
pub struct GpuPointer {
    /// Pointeur brut vers la mémoire GPU
    #[cfg(feature = "gpu")]
    ptr: cust::driver::CUdeviceptr,
    /// Taille du bloc alloué en octets
    pub size: usize,
    /// Identifiant du device propriétaire
    pub device_id: usize,
    /// Flag pour éviter la double libération
    #[allow(dead_code)] // Utilisé uniquement dans le bloc #[cfg(feature = "gpu")]
    freed: AtomicBool,
}

// SAFETY : GpuPointer contient un pointeur brut (CUdeviceptr) qui est un entier non signé.
// Ce pointeur peut être envoyé entre threads car il ne contient pas de state mutable partagé.
// La libération de la mémoire est gérée par le Drop, qui est safe.
unsafe impl Send for GpuPointer {}

impl GpuPointer {
    /// Crée un nouveau pointeur GPU (interne)
    ///
    /// # Sécurité
    ///
    /// Cette fonction est unsafe car elle manipule des pointeurs bruts.
    #[cfg(feature = "gpu")]
    unsafe fn new(ptr: cust::driver::CUdeviceptr, size: usize, device_id: usize) -> Self {
        Self {
            ptr,
            size,
            device_id,
            freed: AtomicBool::new(false),
        }
    }

    /// Retourne le pointeur brut (usage interne)
    #[cfg(feature = "gpu")]
    pub fn raw_ptr(&self) -> cust::driver::CUdeviceptr {
        self.ptr
    }
}

impl Drop for GpuPointer {
    fn drop(&mut self) {
        #[cfg(feature = "gpu")]
        {
            use std::sync::atomic::Ordering;
            // Vérifier si déjà libéré
            if self.freed.load(Ordering::Acquire) {
                return;
            }

            // Marquer comme libéré avant la libération
            self.freed.store(true, Ordering::Release);

            // Libérer la mémoire GPU
            // SAFETY:
            // - self.ptr a été alloué via cuMemAlloc dans allocate_memory()
            // - Le pointeur est validé non-null par la vérification précédente
            // - Le Drop est appelé une seule fois grâce au type system Rust
            unsafe {
                if let Err(e) = cust::driver::cuMemFree(self.ptr) {
                    log::error!(
                        "Échec de la libération de la mémoire GPU {}: {}",
                        self.device_id,
                        e
                    );
                }
            }
        }
    }
}

/// Device GPU
///
/// Représente un device GPU initialisé et prêt à l'emploi.
/// Gère l'initialisation du contexte CUDA et les opérations de base.
pub struct GpuDevice {
    /// Informations sur le device
    info: DeviceInfo,
    /// Contexte CUDA (uniquement avec la feature gpu)
    #[cfg(feature = "gpu")]
    context: cust::context::Context,
    /// Compteur d'allocations actives
    allocation_count: Arc<Mutex<usize>>,
}

impl GpuDevice {
    /// Crée un nouveau device GPU
    ///
    /// # Arguments
    ///
    /// * `device_id` - Identifiant du device à initialiser
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::GpuNotAvailable` si le device n'est pas disponible.
    ///
    /// # Exemple
    ///
    /// ```rust,no_run
    /// use pmg_gpu::GpuDevice;
    ///
    /// let device = GpuDevice::new(0).unwrap();
    /// println!("Device: {}", device.info().name);
    /// ```
    pub fn new(device_id: usize) -> GpuResult<Self> {
        #[cfg(feature = "gpu")]
        {
            // Initialiser le runtime CUDA
            cust::quick_init()
                .map_err(|e| GpuError::CudaError(format!("Initialisation CUDA échouée: {}", e)))?;

            // Obtenir le device
            let device = cust::device::get_device(device_id).map_err(|e| {
                GpuError::CudaError(format!("Device {} non trouvé: {}", device_id, e))
            })?;

            // Obtenir les propriétés
            let props = device
                .get_properties()
                .map_err(|e| GpuError::CudaError(format!("Propriétés inaccessibles: {}", e)))?;

            let info = DeviceInfo {
                id: device_id,
                name: props.name().to_string(),
                compute_capability: props.compute_capability(),
                total_memory: props.total_global_mem(),
                free_memory: props.total_global_mem(), // Approximation initiale
                max_threads_per_block: props.max_threads_per_block(),
                max_blocks_per_grid: props.max_grid_dim_x(),
                warp_size: props.warp_size(),
            };

            // Créer le contexte
            let context = cust::context::Context::new(device)
                .map_err(|e| GpuError::CudaError(format!("Contexte CUDA échoué: {}", e)))?;

            Ok(Self {
                info,
                context,
                allocation_count: Arc::new(Mutex::new(0)),
            })
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mode fallback: simuler un device GPU
            log::info!("Mode fallback CPU pour le device {}", device_id);

            let info = DeviceInfo {
                id: device_id,
                name: format!("GPU virtuel (device {})", device_id),
                compute_capability: (7, 0),
                total_memory: 8 * 1024 * 1024 * 1024, // 8 GB virtuel
                free_memory: 8 * 1024 * 1024 * 1024,
                max_threads_per_block: 1024,
                max_blocks_per_grid: 2147483647,
                warp_size: 32,
            };

            Ok(Self {
                info,
                allocation_count: Arc::new(Mutex::new(0)),
            })
        }
    }

    /// Retourne les informations du device
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Alloue de la mémoire GPU
    ///
    /// # Arguments
    ///
    /// * `size` - Nombre d'octets à allouer
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::AllocationError` si l'allocation échoue.
    ///
    /// # Sécurité
    ///
    /// La mémoire allouée est automatiquement libérée lors de la destruction
    /// du pointeur retourné.
    pub fn allocate(&self, size: usize) -> GpuResult<GpuPointer> {
        if size == 0 {
            return Err(GpuError::AllocationError(
                "Taille d'allocation invalide".to_string(),
            ));
        }

        #[cfg(feature = "gpu")]
        {
            // Allouer la mémoire GPU
            let mut ptr: cust::driver::CUdeviceptr = 0;
            // SAFETY:
            // - size > 0 est vérifié avant l'appel
            // - Le pointeur résultat est validé par cuMemAlloc
            // - Les erreurs sont propagées via le ?
            unsafe {
                cust::driver::cuMemAlloc(&mut ptr, size).map_err(|e| {
                    GpuError::AllocationError(format!("Échec allocation {} octets: {}", size, e))
                })?;
            }

            // Mettre à jour le compteur
            if let Ok(mut count) = self.allocation_count.lock() {
                *count += 1;
            }

            log::debug!(
                "Allocation GPU: {} octets sur device {}",
                size,
                self.info.id
            );

            // SAFETY:
            // - ptr est le résultat valide de cuMemAlloc
            // - size correspond à la taille allouée
            // - device_id est valide (vérifié par get_device())
            unsafe { Ok(GpuPointer::new(ptr, size, self.info.id)) }
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mode fallback: simuler l'allocation
            log::debug!("Simulation allocation GPU: {} octets", size);

            if let Ok(mut count) = self.allocation_count.lock() {
                *count += 1;
            }

            // Retourner un pointeur "virtuel" (0) avec la taille
            // Note: en mode fallback, les opérations memcpy seront simulées
            Ok(GpuPointer {
                size,
                device_id: self.info.id,
                freed: AtomicBool::new(false),
            })
        }
    }

    /// Copie des données vers le GPU (host → device)
    ///
    /// # Arguments
    ///
    /// * `dst` - Pointeur de destination sur le GPU
    /// * `src` - Données source sur le CPU
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::TransferError` si la copie échoue.
    pub fn memcpy_to_device(&self, dst: &GpuPointer, src: &[u8]) -> GpuResult<()> {
        if src.len() > dst.size {
            return Err(GpuError::TransferError(format!(
                "Source ({} octets) dépasse la destination ({} octets)",
                src.len(),
                dst.size
            )));
        }

        #[cfg(feature = "gpu")]
        {
            // SAFETY:
            // - src.len() <= dst.size est vérifié avant l'appel
            // - src est un buffer valide (emprunté)
            // - dst.ptr est un pointeur GPU valide
            unsafe {
                cust::driver::cuMemcpyHtoD(dst.raw_ptr(), src.as_ptr() as *const _, src.len())
                    .map_err(|e| {
                        GpuError::TransferError(format!("Copie host→device échouée: {}", e))
                    })?;
            }
            Ok(())
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mode fallback: simuler la copie
            log::debug!("Simulation copie host→device: {} octets", src.len());
            Ok(())
        }
    }

    /// Copie des données depuis le GPU (device → host)
    ///
    /// # Arguments
    ///
    /// * `dst` - Buffer de destination sur le CPU
    /// * `src` - Pointeur source sur le GPU
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::TransferError` si la copie échoue.
    pub fn memcpy_from_device(&self, dst: &mut [u8], src: &GpuPointer) -> GpuResult<()> {
        if dst.len() > src.size {
            return Err(GpuError::TransferError(format!(
                "Destination ({} octets) dépasse la source ({} octets)",
                dst.len(),
                src.size
            )));
        }

        #[cfg(feature = "gpu")]
        {
            // SAFETY:
            // - dst.len() <= src.size est vérifié avant l'appel
            // - dst est un buffer mutable valide
            // - src.ptr est un pointeur GPU valide
            unsafe {
                cust::driver::cuMemcpyDtoH(dst.as_mut_ptr() as *mut _, src.raw_ptr(), dst.len())
                    .map_err(|e| {
                        GpuError::TransferError(format!("Copie device→host échouée: {}", e))
                    })?;
            }
            Ok(())
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Mode fallback: simuler la copie
            log::debug!("Simulation copie device→host: {} octets", dst.len());
            // Remplir avec des données simulées
            for byte in dst.iter_mut() {
                *byte = 0;
            }
            Ok(())
        }
    }

    /// Décrémente le compteur d'allocations actives
    pub(crate) fn decrement_allocation_count(&self) {
        if let Ok(mut count) = self.allocation_count.lock() {
            *count = count.saturating_sub(1);
        }
    }

    /// Retourne le nombre d'allocations actives
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.lock().map(|c| *c).unwrap_or(0)
    }

    /// Vérifie si le device est disponible
    pub fn is_available(&self) -> bool {
        true // Le device est toujours disponible s'il a été créé
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("id", &self.info.id)
            .field("name", &self.info.name)
            .field("memory", &self.info.total_memory)
            .finish()
    }
}

/// Gestionnaire de devices GPU
///
/// Fournit des méthodes utilitaires pour la gestion de multiples devices.
pub struct DeviceManager {
    devices: HashMap<usize, Arc<GpuDevice>>,
}

impl DeviceManager {
    /// Crée un nouveau gestionnaire de devices
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    /// Ajoute un device au gestionnaire
    pub fn add_device(&mut self, device: GpuDevice) {
        let id = device.info().id;
        self.devices.insert(id, Arc::new(device));
    }

    /// Retourne un device par son identifiant
    pub fn get_device(&self, id: usize) -> Option<Arc<GpuDevice>> {
        self.devices.get(&id).cloned()
    }

    /// Retourne le nombre de devices gérés
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Retourne les informations de tous les devices
    pub fn devices_info(&self) -> Vec<DeviceInfo> {
        self.devices.values().map(|d| d.info().clone()).collect()
    }

    /// Sélectionne le device avec le plus de mémoire libre
    pub fn select_best_device(&self) -> Option<Arc<GpuDevice>> {
        self.devices
            .values()
            .max_by_key(|d| d.info().free_memory)
            .cloned()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_default() {
        let info = DeviceInfo::default();
        assert_eq!(info.id, 0);
        assert!(info.total_memory > 0);
    }

    #[test]
    fn test_gpu_device_creation() {
        // Ce test vérifie que la création ne panic pas
        let result = GpuDevice::new(0);
        assert!(result.is_ok());

        let device = result.unwrap();
        assert!(device.info().total_memory > 0);
        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    fn test_device_manager() {
        let mut manager = DeviceManager::new();
        assert_eq!(manager.device_count(), 0);

        let device = GpuDevice::new(0).unwrap();
        manager.add_device(device);

        assert_eq!(manager.device_count(), 1);
        assert!(manager.get_device(0).is_some());
        assert!(manager.select_best_device().is_some());
    }

    #[test]
    fn test_gpu_pointer_simulation() {
        let device = GpuDevice::new(0).unwrap();
        let pointer = device.allocate(1024).unwrap();

        assert_eq!(pointer.size, 1024);
        assert_eq!(device.allocation_count(), 1);
    }
}
