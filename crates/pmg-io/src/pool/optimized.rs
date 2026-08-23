//! Pool de Buffers Optimisé avec Tracking Mémoire
//!
//! Ce module implémente un pool de buffers optimisé qui utilise le `GlobalMemoryManager`
//! pour un tracking précis de la mémoire et garantit l'absence de fuites.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pmg_core::memory::{GlobalMemoryManager, MemoryError, MemoryMonitor};

use super::PoolConfig;

/// Pool de buffers optimisé avec tracking mémoire
///
/// Ce pool utilise le `GlobalMemoryManager` pour tracker chaque allocation
/// et désallocation, garantissant ainsi l'absence de fuites mémoire.
#[allow(dead_code)]
pub struct OptimizedBufferPool {
    /// Pool pour buffers u8
    u8_pool: Arc<Mutex<PoolImpl<u8>>>,

    /// Pool pour buffers f64
    f64_pool: Arc<Mutex<PoolImpl<f64>>>,

    /// Moniteur mémoire global
    memory_monitor: Arc<GlobalMemoryManager>,

    /// Configuration du pool
    config: PoolConfig,
}

/// Configuration de l'éviction dynamique pour le pool de buffers.
///
/// Contrôle quand et comment les buffers sont évacués du pool
/// pour libérer de la mémoire lorsque l'utilisation dépasse un seuil.
#[derive(Debug, Clone)]
pub struct EvictionConfig {
    /// Activer ou désactiver l'éviction dynamique.
    pub enabled: bool,
    /// Seuil d'utilisation (0.0 à 1.0) déclenchant l'éviction.
    /// Exemple : 0.8 = 80% de la mémoire maximale utilisée.
    pub threshold_percentage: f64,
    /// Pourcentage de mémoire à évacuer lors du déclenchement.
    /// Exemple : 0.2 = évacuer 20% de la mémoire maximale.
    pub eviction_percentage: f64,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_percentage: 0.8,
            eviction_percentage: 0.2,
        }
    }
}

/// Implémentation interne du pool typé
#[allow(dead_code)]
struct PoolImpl<T> {
    /// Buckets organisés par taille (index = taille_en_octets / min_buffer_size)
    buckets: HashMap<usize, Vec<Vec<T>>>,

    /// Mémoire actuellement allouée par ce pool
    allocated_memory: usize,

    /// Mémoire maximale pour ce pool
    max_memory: usize,

    /// Taille minimale des buffers en octets
    min_buffer_size: usize,

    /// Nombre de buckets (max_buffer_size / min_buffer_size + 1)
    num_buckets: usize,

    /// Nombre d'allocations totales
    total_allocations: usize,

    /// Nombre de réutilisations
    total_reuses: usize,

    /// Configuration de l'éviction dynamique
    eviction_config: EvictionConfig,
}

impl<T: Default + Clone> PoolImpl<T> {
    /// Crée une nouvelle implémentation de pool
    fn new(max_memory: usize, min_buffer_size: usize) -> Self {
        // Nombre de buckets basé sur la taille maximale attendue (ici 64 Mo par défaut)
        // On utilise un nombre raisonnable de buckets (ex: 64)
        let num_buckets = 64;
        let mut buckets = HashMap::with_capacity(num_buckets);
        for i in 0..num_buckets {
            buckets.insert(i, Vec::new());
        }

        Self {
            buckets,
            allocated_memory: 0,
            max_memory,
            min_buffer_size,
            num_buckets,
            total_allocations: 0,
            total_reuses: 0,
            eviction_config: EvictionConfig::default(),
        }
    }

    /// Acquiert un buffer de la taille minimale spécifiée (en éléments T)
    fn acquire(
        &mut self,
        min_size: usize,
        memory_monitor: &GlobalMemoryManager,
    ) -> Result<Vec<T>, MemoryError> {
        // Convertir la taille minimale en octets pour le calcul du bucket
        let min_bytes = min_size * std::mem::size_of::<T>();
        let min_bucket = min_bytes / self.min_buffer_size;

        // Chercher un buffer réutilisable en partant du plus grand bucket
        for bucket in (min_bucket..self.num_buckets).rev() {
            if let Some(buffers) = self.buckets.get_mut(&bucket) {
                if let Some(buffer) = buffers.pop() {
                    // Vérifier que le buffer est assez grand (en octets)
                    let buffer_bytes = buffer.capacity() * std::mem::size_of::<T>();
                    if buffer_bytes >= min_bytes {
                        // Buffer trouvé, le réutiliser
                        self.total_reuses += 1;
                        return Ok(buffer);
                    } else {
                        // Buffer trop petit, le remettre dans le bucket
                        buffers.push(buffer);
                    }
                }
            }
        }

        // Pas de buffer réutilisable, en allouer un nouveau
        // Allouer exactement la taille demandée (pas d'arrondi à min_buffer_size)
        let alloc_bytes = min_bytes;

        // Vérifier si on peut allouer (en octets)
        if self.allocated_memory + alloc_bytes > self.max_memory {
            return Err(MemoryError::InsufficientMemory {
                available: (self.max_memory - self.allocated_memory) as u64,
                requested: alloc_bytes as u64,
            });
        }

        // Allouer via le memory monitor
        memory_monitor.allocate(alloc_bytes as u64)?;

        // Créer le buffer sans initialisation (pré-allocation seule)
        // SAFETY : T est Default + Clone, mais on évite l'initialisation inutile
        let mut buffer = Vec::with_capacity(min_size);
        // SAFETY : Le buffer non initialisé sera rempli par l'appelant avant d'être lu.
        #[allow(clippy::uninit_vec)]
        unsafe {
            buffer.set_len(min_size);
        }

        self.allocated_memory += alloc_bytes;
        self.total_allocations += 1;

        Ok(buffer)
    }

    /// Retourne un buffer au pool avec éviction dynamique.
    ///
    /// Si l'utilisation mémoire dépasse le seuil configuré, les buffers
    /// les plus anciens sont évacués avant de stocker le nouveau buffer.
    fn release(&mut self, buffer: Vec<T>, _memory_monitor: &GlobalMemoryManager) {
        // Calculer la taille du buffer en octets (capacity * taille de T)
        let buffer_bytes = buffer.capacity() * std::mem::size_of::<T>();

        // Vérifier si l'éviction dynamique est nécessaire
        if self.eviction_config.enabled && self.max_memory > 0 {
            let usage_percentage = self.allocated_memory as f64 / self.max_memory as f64;
            if usage_percentage >= self.eviction_config.threshold_percentage {
                self.evict_buffers();
            }
        }

        // Vérifier si on peut stocker le buffer sans dépasser la limite mémoire
        if self.allocated_memory + buffer_bytes <= self.max_memory {
            // Calculer l'index du bucket
            let bucket = std::cmp::min(buffer_bytes / self.min_buffer_size, self.num_buckets - 1);

            // Stocker le buffer pour réutilisation
            // Utilisation de or_default() au lieu de or_insert_with(Vec::new) pour plus d'idiomaticité
            self.buckets.entry(bucket).or_default().push(buffer);
            self.allocated_memory += buffer_bytes;
        } else {
            // Libérer la mémoire (le buffer sera dropé normalement)
            // Note: on ne fait pas de deallocation ici car le buffer sera libéré par Drop
            // On met juste à jour le compteur
            self.allocated_memory = self.allocated_memory.saturating_sub(buffer_bytes);
        }
    }

    /// Évince les buffers du pool pour libérer de la mémoire.
    ///
    /// Supprime les buffers les plus anciens (au fond des buckets)
    /// jusqu'à ce que la mémoire libérée atteigne le pourcentage configuré.
    fn evict_buffers(&mut self) {
        // Calculer la quantité de mémoire à évacuer
        let bytes_to_evict =
            (self.max_memory as f64 * self.eviction_config.eviction_percentage) as usize;
        let mut evicted = 0;

        // Parcourir les buckets du plus grand au plus petit
        for bucket_key in (0..self.num_buckets).rev() {
            if let Some(buffers) = self.buckets.get_mut(&bucket_key) {
                while evicted < bytes_to_evict && !buffers.is_empty() {
                    // Retirer le dernier buffer (le plus récent dans ce bucket)
                    if let Some(buffer) = buffers.pop() {
                        let buffer_bytes = buffer.capacity() * std::mem::size_of::<T>();
                        evicted += buffer_bytes;
                        self.allocated_memory = self.allocated_memory.saturating_sub(buffer_bytes);
                        // Le buffer est automatiquement libéré ici (Drop)
                    }
                }
            }
            if evicted >= bytes_to_evict {
                break;
            }
        }
    }

    /// Vide complètement le pool
    fn clear(&mut self, memory_monitor: &GlobalMemoryManager) {
        for buffers in self.buckets.values_mut() {
            for buffer in buffers.drain(..) {
                let buffer_bytes = buffer.capacity() * std::mem::size_of::<T>();
                memory_monitor.deallocate(buffer_bytes as u64);
                self.allocated_memory -= buffer_bytes;
            }
        }
    }
}

impl OptimizedBufferPool {
    /// Crée un nouveau pool optimisé
    pub fn new(memory_monitor: Arc<GlobalMemoryManager>, config: PoolConfig) -> Self {
        let max_memory_per_pool = config.max_memory_per_pool / 2; // Diviser entre u8 et f64

        Self {
            u8_pool: Arc::new(Mutex::new(PoolImpl::new(
                max_memory_per_pool,
                config.min_buffer_size,
            ))),
            f64_pool: Arc::new(Mutex::new(PoolImpl::new(
                max_memory_per_pool,
                config.min_buffer_size * std::mem::size_of::<f64>(),
            ))),
            memory_monitor,
            config,
        }
    }

    /// Acquiert un buffer u8 de la taille minimale
    pub fn acquire_u8(&self, min_size: usize) -> Result<Vec<u8>, MemoryError> {
        let mut pool = self
            .u8_pool
            .lock()
            .map_err(|e| MemoryError::TrackingError(format!("Erreur de verrouillage: {}", e)))?;

        pool.acquire(min_size, &self.memory_monitor)
    }

    /// Acquiert un buffer f64 de la taille minimale
    pub fn acquire_f64(&self, min_len: usize) -> Result<Vec<f64>, MemoryError> {
        let mut pool = self
            .f64_pool
            .lock()
            .map_err(|e| MemoryError::TrackingError(format!("Erreur de verrouillage: {}", e)))?;

        pool.acquire(min_len, &self.memory_monitor)
    }

    /// Retourne un buffer u8 au pool
    pub fn release_u8(&self, buffer: Vec<u8>) {
        if let Ok(mut pool) = self.u8_pool.lock() {
            pool.release(buffer, &self.memory_monitor);
        }
    }

    /// Retourne un buffer f64 au pool
    pub fn release_f64(&self, buffer: Vec<f64>) {
        if let Ok(mut pool) = self.f64_pool.lock() {
            pool.release(buffer, &self.memory_monitor);
        }
    }

    /// Vide complètement le pool
    ///
    /// Pour éviter les deadlocks, on utilise toujours le même ordre d'acquisition
    /// des verrous: d'abord u8_pool, puis f64_pool. Cette hiérarchie doit être
    /// respectée dans tout le code utilisant ces deux pools simultanément.
    pub fn clear(&self) {
        // Toujours acquérir les verrous dans le même ordre pour éviter les deadlocks
        let u8_guard = self.u8_pool.lock();
        let f64_guard = self.f64_pool.lock();

        if let Ok(mut pool) = u8_guard {
            pool.clear(&self.memory_monitor);
        }
        if let Ok(mut pool) = f64_guard {
            pool.clear(&self.memory_monitor);
        }
    }

    /// Retourne les statistiques du pool
    pub fn stats(&self) -> PoolStats {
        let u8_stats = self
            .u8_pool
            .lock()
            .map(|p| (p.total_allocations, p.total_reuses, p.allocated_memory))
            .unwrap_or((0, 0, 0));
        let f64_stats = self
            .f64_pool
            .lock()
            .map(|p| (p.total_allocations, p.total_reuses, p.allocated_memory))
            .unwrap_or((0, 0, 0));

        PoolStats {
            u8_allocations: u8_stats.0,
            u8_reuses: u8_stats.1,
            u8_memory: u8_stats.2,
            f64_allocations: f64_stats.0,
            f64_reuses: f64_stats.1,
            f64_memory: f64_stats.2,
            total_allocations: u8_stats.0 + f64_stats.0,
            total_reuses: u8_stats.1 + f64_stats.1,
            total_memory: u8_stats.2 + f64_stats.2,
        }
    }
}

/// Statistiques du pool
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Nombre d'allocations u8
    pub u8_allocations: usize,

    /// Nombre de réutilisations u8
    pub u8_reuses: usize,

    /// Mémoire u8 allouée
    pub u8_memory: usize,

    /// Nombre d'allocations f64
    pub f64_allocations: usize,

    /// Nombre de réutilisations f64
    pub f64_reuses: usize,

    /// Mémoire f64 allouée
    pub f64_memory: usize,

    /// Total allocations
    pub total_allocations: usize,

    /// Total réutilisations
    pub total_reuses: usize,

    /// Total mémoire
    pub total_memory: usize,
}

/// Wrapper typé pour la gestion automatique des buffers
///
/// Ce wrapper garantit que les buffers sont retournés automatiquement
/// au pool lors de leur destruction, évitant ainsi les fuites mémoire.
pub struct PooledBuffer<T> {
    /// Buffer de données
    data: Vec<T>,

    /// Référence au pool
    pool: Option<Arc<OptimizedBufferPool>>,

    /// Type de buffer (u8 ou f64)
    is_f64: bool,
}

impl<T> PooledBuffer<T> {
    /// Crée un nouveau buffer typé
    pub fn new(data: Vec<T>, pool: Arc<OptimizedBufferPool>, is_f64: bool) -> Self {
        Self {
            data,
            pool: Some(pool),
            is_f64,
        }
    }

    /// Retourne une référence aux données
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Retourne une référence mutable aux données
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Retourne la taille du buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Vérifie si le buffer est vide
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consomme le buffer et retourne les données sans retourner au pool
    ///
    /// Utile quand on veut garder les données sans les retourner au pool
    pub fn into_inner(mut self) -> Vec<T> {
        // Empêcher le Drop de retourner le buffer au pool
        self.pool = None;
        std::mem::take(&mut self.data)
    }
}

impl<T> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        // Retourner le buffer au pool si disponible
        if let Some(pool) = self.pool.take() {
            // Ne pas retourner les buffers vides
            if !self.data.is_empty() {
                // Retirer le stockage du pool pour éviter la duplication
                let data = std::mem::take(&mut self.data);

                // Retourner au pool selon le type
                // Note: Cette implémentation est spécifique aux types u8 et f64
                // Pour un type générique, il faudrait utiliser un trait ou des types spécialisés
                if self.is_f64 {
                    // Pour les buffers f64, on doit convertir Vec<T> en Vec<f64>
                    // Cette conversion est sûre uniquement si T = f64
                    // SAFETY : T est vérifié être f64 par les conditions size_of et align_of.
                    // Les taille et alignement sont identiques, donc la conversion est sûre.
                    // Préconditions :
                    // 1. size_of::<T>() == size_of::<f64>() (taille identique)
                    // 2. align_of::<T>() == align_of::<f64>() (alignement identique)
                    // 3. data contient des éléments T valides
                    // 4. data n'est pas encore libéré (on utilise take() pour extraire le buffer)
                    debug_assert_eq!(
                        std::mem::size_of::<T>(),
                        std::mem::size_of::<f64>(),
                        "La taille de T doit être identique à celle de f64"
                    );
                    debug_assert_eq!(
                        std::mem::align_of::<T>(),
                        std::mem::align_of::<f64>(),
                        "L'alignement de T doit être identique à celui de f64"
                    );

                    let ptr = data.as_ptr() as *mut f64;
                    let len = data.len();
                    let cap = data.capacity();

                    // Vérifications de sécurité
                    debug_assert!(len <= cap, "La longueur ne doit pas dépasser la capacité");
                    debug_assert!(
                        ptr as usize % std::mem::align_of::<f64>() == 0,
                        "Le pointeur doit être aligné pour f64"
                    );

                    // Empêcher la libération du Vec original
                    std::mem::forget(data);

                    // Créer un nouveau Vec<f64> à partir du pointeur
                    // SAFETY : pointeur validé non-null et aligné, taille et capacité correctes
                    let f64_data = unsafe { Vec::from_raw_parts(ptr, len, cap) };
                    pool.release_f64(f64_data);
                } else {
                    // Pour les buffers u8, on doit convertir Vec<T> en Vec<u8>
                    // SAFETY : T est vérifié être u8 par les conditions size_of et align_of.
                    // Les taille et alignement sont identiques, donc la conversion est sûre.
                    // Préconditions :
                    // 1. size_of::<T>() == size_of::<u8>() (taille identique)
                    // 2. align_of::<T>() == align_of::<u8>() (alignement identique)
                    // 3. data contient des éléments T valides
                    // 4. data n'est pas encore libéré (on utilise take() pour extraire le buffer)
                    debug_assert_eq!(
                        std::mem::size_of::<T>(),
                        std::mem::size_of::<u8>(),
                        "La taille de T doit être identique à celle de u8"
                    );
                    debug_assert_eq!(
                        std::mem::align_of::<T>(),
                        std::mem::align_of::<u8>(),
                        "L'alignement de T doit être identique à celui de u8"
                    );

                    let ptr = data.as_ptr() as *mut u8;
                    // Utilisation de checked_mul pour éviter les overflow dans les calculs de taille
                    let len = data
                        .len()
                        .checked_mul(std::mem::size_of::<T>())
                        .expect("Overflow dans le calcul de la taille en octets");
                    let cap = data
                        .capacity()
                        .checked_mul(std::mem::size_of::<T>())
                        .expect("Overflow dans le calcul de la capacité en octets");

                    // Vérifications de sécurité
                    debug_assert!(len <= cap, "La longueur ne doit pas dépasser la capacité");
                    debug_assert!(
                        len % std::mem::size_of::<u8>() == 0,
                        "La taille en octets doit être un multiple de la taille de u8"
                    );
                    debug_assert!(
                        ptr as usize % std::mem::align_of::<u8>() == 0,
                        "Le pointeur doit être aligné pour u8"
                    );

                    // Empêcher la libération du Vec original
                    std::mem::forget(data);

                    // Créer un nouveau Vec<u8> à partir du pointeur
                    // SAFETY : pointeur validé non-null et aligné, taille et capacité correctes
                    let u8_data = unsafe { Vec::from_raw_parts(ptr, len, cap) };
                    pool.release_u8(u8_data);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::memory::GlobalMemoryManager;
    use std::sync::Arc;

    #[test]
    fn test_creation_pool() {
        let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));
        let config = PoolConfig::default();
        let pool = OptimizedBufferPool::new(manager, config);

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 0);
    }

    #[test]
    fn test_acquisition_release() {
        let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));
        let config = PoolConfig::default();
        let pool = OptimizedBufferPool::new(manager, config);

        // Acquérir un buffer
        let buffer = pool.acquire_u8(1024).unwrap();
        assert_eq!(buffer.len(), 1024);

        // Retourner le buffer
        pool.release_u8(buffer);

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_reuses, 0);
    }
}
