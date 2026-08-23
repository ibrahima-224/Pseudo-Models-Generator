//! Allocations mémoire GPU
//!
//! Ce module implémente l'allocateur principal pour la mémoire GPU,
//! incluant la gestion des allocations et le suivi des statistiques.

use crate::device::{GpuDevice, GpuPointer};
use crate::error::{GpuError, GpuResult};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::stats::AllocationStats;

/// Allocations mémoire GPU
pub struct GpuAllocator {
    /// Device GPU associé à cet allocateur
    device: Arc<GpuDevice>,
    /// Carte des allocations actives indexées par device_id
    active_allocations: Mutex<BTreeMap<usize, AllocationInfo>>,
    /// Statistiques d'allocation protégées par un Mutex
    stats: Mutex<AllocationStats>,
}

/// Informations sur une allocation
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AllocationInfo {
    /// Taille de l'allocation en octets
    size: usize,
    /// Horodatage de création de l'allocation
    created_at: std::time::Instant,
    /// Étiquette optionnelle pour le diagnostic
    label: Option<String>,
}

impl GpuAllocator {
    /// Crée un nouvel allocateur pour un device
    ///
    /// # Arguments
    ///
    /// * `device` - Device GPU à utiliser
    ///
    /// # Exemple
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use pmg_gpu::{GpuDevice, GpuAllocator};
    ///
    /// let device = Arc::new(GpuDevice::new(0).unwrap());
    /// let allocator = GpuAllocator::new(device).unwrap();
    /// ```
    pub fn new(device: Arc<GpuDevice>) -> GpuResult<Self> {
        Ok(Self {
            device,
            active_allocations: Mutex::new(BTreeMap::new()),
            stats: Mutex::new(AllocationStats::default()),
        })
    }

    /// Alloue de la mémoire GPU
    ///
    /// # Arguments
    ///
    /// * `size` - Taille en octets à allouer
    /// * `label` - Étiquette optionnelle pour le diagnostic
    ///
    /// # Retour
    ///
    /// Un bloc de mémoire alloué encapsulé dans un `GpuAllocatorBlock`.
    ///
    /// # Erreurs
    ///
    /// Retourne `GpuError::AllocationError` si la taille est invalide.
    pub fn allocate(
        self: Arc<Self>,
        size: usize,
        label: Option<&str>,
    ) -> GpuResult<GpuAllocatorBlock> {
        if size == 0 {
            return Err(GpuError::AllocationError(
                "Taille d'allocation invalide".to_string(),
            ));
        }

        // Allouer la mémoire via le device
        let pointer = self.device.allocate(size)?;

        // Enregistrer l'allocation
        let info = AllocationInfo {
            size,
            created_at: std::time::Instant::now(),
            label: label.map(|s| s.to_string()),
        };

        if let Ok(mut allocations) = self.active_allocations.lock() {
            allocations.insert(pointer.device_id, info);
        }

        // Mettre à jour les statistiques
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocations += 1;
            stats.total_bytes_allocated += size as u64;
            stats.active_allocations += 1;
            stats.active_bytes += size;
            stats.max_allocation_size = stats.max_allocation_size.max(size);
            if stats.min_allocation_size == 0 || size < stats.min_allocation_size {
                stats.min_allocation_size = size;
            }
        }

        log::debug!("Allocation GPU: {} octets (label: {:?})", size, label);

        Ok(GpuAllocatorBlock {
            pointer,
            allocator: self,
        })
    }

    /// Libère une allocation
    ///
    /// # Arguments
    ///
    /// * `pointer` - Pointeur à libérer
    /// * `size` - Taille de l'allocation (pour les statistiques)
    pub(crate) fn release(&self, pointer: &GpuPointer, size: usize) {
        // Supprimer du suivi des allocations
        if let Ok(mut allocations) = self.active_allocations.lock() {
            allocations.remove(&pointer.device_id);
        }

        // Mettre à jour les statistiques
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_deallocations += 1;
            stats.total_bytes_deallocated += size as u64;
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
            stats.active_bytes = stats.active_bytes.saturating_sub(size);
        }

        log::debug!("Libération GPU: {} octets", size);
    }

    /// Retourne les statistiques d'allocation
    pub fn stats(&self) -> AllocationStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Retourne le device associé
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }

    /// Vérifie la cohérence des allocations
    pub fn validate(&self) -> GpuResult<()> {
        let stats = self.stats();

        if stats.active_allocations != self.active_allocations.lock().map(|a| a.len()).unwrap_or(0)
        {
            return Err(GpuError::InternalError(
                "Incohérence entre statistiques et allocations actives".to_string(),
            ));
        }

        Ok(())
    }
}

/// Bloc de mémoire alloué
///
/// Représente un bloc de mémoire alloué sur le GPU.
/// La mémoire est automatiquement libérée lors de la destruction.
///
/// **ATTENTION - Problème de sécurité connu** :
/// Le champ `allocator` utilise un pointeur brut (*const GpuAllocator) au lieu
/// d'un Arc<GpuAllocator>. Cela pose des risques de mémoire en cas de libération
/// prématurée de l'allocateur. Ce point doit être corrigé dans une refonte
/// future de l'architecture (voir TODO ci-dessous).
pub struct GpuAllocatorBlock {
    /// Pointeur sous-jacent
    pub(crate) pointer: GpuPointer,
    /// Référence à l'allocateur (Arc pour la sécurité mémoire)
    pub(crate) allocator: Arc<GpuAllocator>,
}

impl GpuAllocatorBlock {
    /// Retourne le pointeur GPU
    pub fn pointer(&self) -> &GpuPointer {
        &self.pointer
    }

    /// Retourne la taille du bloc en octets
    pub fn size(&self) -> usize {
        self.pointer.size
    }

    /// Copie des données vers le GPU
    pub fn copy_to_device(&self, data: &[u8]) -> GpuResult<()> {
        self.allocator
            .device()
            .memcpy_to_device(&self.pointer, data)
    }

    /// Copie des données depuis le GPU
    pub fn copy_from_device(&self, data: &mut [u8]) -> GpuResult<()> {
        self.allocator
            .device()
            .memcpy_from_device(data, &self.pointer)
    }
}

impl Drop for GpuAllocatorBlock {
    fn drop(&mut self) {
        // Libérer l'allocation via l'allocateur
        self.allocator.release(&self.pointer, self.pointer.size);

        // Décrémenter le compteur d'allocations actives du device
        self.allocator.device().decrement_allocation_count();
    }
}

impl std::fmt::Debug for GpuAllocatorBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuAllocatorBlock")
            .field("size", &self.pointer.size)
            .field("device_id", &self.pointer.device_id)
            .finish()
    }
}
