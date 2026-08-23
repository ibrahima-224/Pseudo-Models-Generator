// Copyright (C) 2024 PMG Contributors
// This file is part of PMG (Pseudo-Model Generator).
//
// PMG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PMG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PMG.  If not, see <https://www.gnu.org/licenses/>.

//! # Pool de Buffers Unifié Thread-Safe
//!
//! Ce module implémente le pool de buffers principal, partagé et thread-safe,
//! capable de gérer des buffers de différentes tailles (1 Mo à 64 Mo).
//!
//! ## Architecture
//!
//! Le pool utilise `Arc<Mutex<BufferPoolImpl>>` pour le partage thread-safe :
//! - `Arc` : partage du pool entre threads
//! - `Mutex` : protection concurrente des accès
//!
//! ## Gestion mémoire
//!
//! Le pool respecte une limite de mémoire maximale configurable. Lorsque la
//! limite est atteinte, les buffers retournés sont libérés au lieu d'être
//! stockés, assurant une consommation mémoire bornée.

use std::cmp::min;
use std::sync::{Arc, Mutex};

use super::metrics::PoolMetrics;
use super::PoolConfig;

/// Pool de buffers réutilisables et thread-safe.
///
/// Ce pool optimise les allocations mémoire en réutilisant les buffers
/// des tailles courantes. Il est partageable entre threads grâce à
/// `Arc<Mutex<...>>`.
///
/// # Exemple
///
/// ```rust,ignore
/// use pmg_io::pool::{UnifiedBufferPool, PoolConfig};
///
/// let pool = UnifiedBufferPool::new(PoolConfig::default());
/// let mut buf = pool.acquire_u8(1024);
/// buf.extend_from_slice(b"données");
/// // buf est automatiquement retourné au poolwhen dropped
/// ```
#[derive(Clone)]
pub struct UnifiedBufferPool {
    /// Pool interne partagé entre threads (thread-safe).
    inner: Arc<Mutex<BufferPoolImpl>>,
}

/// Implémentation interne du pool de buffers.
///
/// Cette structure contient l'état mutable du pool et n'est pas
/// directement accessible depuis l'extérieur.
struct BufferPoolImpl {
    /// Buffers disponibles classés par index de taille.
    /// L'index est calculé comme `taille / min_buffer_size`.
    pools: Vec<Vec<Vec<u8>>>,

    /// Mémoire totale actuellement allouée dans le pool (en octets).
    total_allocated: usize,

    /// Mémoire maximale autorisée pour le pool (en octets).
    max_pool_memory: usize,

    /// Taille minimale d'un buffer (en octets).
    min_buffer_size: usize,

    /// Nombre de buckets de tailles (max_buffer_size / min_buffer_size + 1).
    num_buckets: usize,

    /// Métriques d'utilisation du pool.
    metrics: PoolMetrics,

    /// Indicateur d'activation des métriques.
    enable_metrics: bool,
}

impl UnifiedBufferPool {
    /// Crée un nouveau pool de buffers unifié avec la configuration spécifiée.
    ///
    /// # Paramètres
    /// - `config` : configuration du pool (limites mémoire, tailles de buffers)
    ///
    /// # Retourne
    /// Un pool de buffers prêt à l'emploi, partageable entre threads.
    pub fn new(config: PoolConfig) -> Self {
        // Valide la configuration
        config
            .validate()
            .expect("Configuration du pool de buffers invalide");

        let num_buckets = config.max_buffer_size / config.min_buffer_size + 1;
        let pools = vec![Vec::new(); num_buckets];

        let inner = BufferPoolImpl {
            pools,
            total_allocated: 0,
            max_pool_memory: config.max_memory_per_pool,
            min_buffer_size: config.min_buffer_size,
            num_buckets,
            metrics: PoolMetrics::new(),
            enable_metrics: config.enable_metrics,
        };

        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Acquiert un buffer u8 de la taille minimale spécifiée.
    ///
    /// Si un buffer réutilisable est disponible dans le pool, il est retourné.
    /// Sinon, un nouveau buffer est alloué.
    ///
    /// # Paramètres
    /// - `min_size` : taille minimale requise pour le buffer (en octets)
    ///
    /// # Retourne
    /// Un `Vec<u8>` de capacité au moins `min_size`.
    pub fn acquire_u8(&self, min_size: usize) -> Vec<u8> {
        let mut inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.acquire_buffer(min_size)
    }

    /// Acquiert un buffer f64 de la taille minimale spécifiée (en éléments).
    ///
    /// Le buffer retourné est un `Vec<f64>` avec une capacité suffisante
    /// pour contenir au moins `min_len` éléments f64.
    ///
    /// # Paramètres
    /// - `min_len` : nombre minimal d'éléments f64 requis
    ///
    /// # Retourne
    /// Un `Vec<f64>` de capacité au moins `min_len`.
    pub fn acquire_f64(&self, min_len: usize) -> Vec<f64> {
        // Conversion : min_len éléments f64 = min_len * 8 octets
        // Utilisation de checked_mul pour éviter les overflow
        let min_bytes = min_len
            .checked_mul(std::mem::size_of::<f64>())
            .expect("Overflow dans le calcul de la taille en octets pour f64");
        let mut byte_buffer = self.acquire_u8(min_bytes);

        // Calculer la capacité en éléments f64
        let capacity_f64 = byte_buffer.capacity() / std::mem::size_of::<f64>();
        let len_f64 = byte_buffer.len() / std::mem::size_of::<f64>();

        // SAFETY :
        // - byte_buffer a été alloué avec une taille multiple de size_of::<f64>()
        // - L'alignement de u8 est inférieur ou égal à celui de f64 (garanti par la layout rule)
        // - On utilise forget() pour éviter la double libération
        // - Le Vec<f64> résultant a le même pointeur, len et capacity (en éléments)
        let ptr = byte_buffer.as_mut_ptr() as *mut f64;

        // Empêcher la deallocation du buffer u8 original
        std::mem::forget(byte_buffer);

        // Retourner directement Vec<f64> sans copie
        unsafe { Vec::from_raw_parts(ptr, len_f64, capacity_f64) }
    }

    /// Remet un buffer u8 dans le pool pour réutilisation future.
    ///
    /// Si le pool a atteint sa limite mémoire, le buffer est libéré
    /// au lieu d'être stocké.
    ///
    /// # Paramètres
    /// - `buffer` : buffer à remettre dans le pool
    pub fn release_u8(&self, buffer: Vec<u8>) {
        let mut inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.release_buffer(buffer);
    }

    /// Remet un buffer f64 dans le pool pour réutilisation future.
    ///
    /// # Paramètres
    /// - `buffer` : buffer f64 à remettre dans le pool
    pub fn release_f64(&self, buffer: Vec<f64>) {
        // Si le buffer est vide, rien à faire
        if buffer.is_empty() {
            return;
        }

        // Calculer la taille en octets
        let len_bytes = buffer.len() * std::mem::size_of::<f64>();
        let cap_bytes = buffer.capacity() * std::mem::size_of::<f64>();

        // SAFETY :
        // - buffer contient des f64 valides (garanti par l'appelant)
        // - La taille en octets est correctement calculée (len * size_of)
        // - L'alignement de f64 est supérieur ou égal à celui de u8 (garanti par la layout rule)
        // - On utilise forget() pour éviter la double libération
        // - Le Vec<u8> résultant a le même pointeur, len (en octets) et capacity (en octets)
        let ptr = buffer.as_ptr() as *mut u8;

        // Empêcher la deallocation du buffer f64 original
        std::mem::forget(buffer);

        // Retourner Vec<u8> sans copie
        let byte_buf = unsafe { Vec::from_raw_parts(ptr, len_bytes, cap_bytes) };

        let mut inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.release_buffer(byte_buf);
    }

    /// Acquiert un `PooledBuffer<u8>` avec remise automatique au pool.
    ///
    /// Cette méthode est un raccourci pour obtenir un buffer typé
    /// qui sera automatiquement retourné au pool when dropped.
    ///
    /// # Paramètres
    /// - `min_size` : taille minimale requise pour le buffer (en octets)
    ///
    /// # Retourne
    /// Un `PooledBuffer<u8>` qui sera automatiquement retourné au pool.
    pub fn acquire_pooled_u8(&self, min_size: usize) -> super::typed_pool::PooledBuffer<u8> {
        let buffer = self.acquire_u8(min_size);
        super::typed_pool::PooledBuffer::new_u8(buffer, self.clone())
    }

    /// Acquiert un `PooledBuffer<f64>` avec remise automatique au pool.
    ///
    /// # Paramètres
    /// - `min_len` : nombre minimal d'éléments f64 requis
    ///
    /// # Retourne
    /// Un `PooledBuffer<f64>` qui sera automatiquement retourné au pool.
    pub fn acquire_pooled_f64(&self, min_len: usize) -> super::typed_pool::PooledBuffer<f64> {
        let buffer = self.acquire_f64(min_len);
        super::typed_pool::PooledBuffer::new_f64(buffer, self.clone())
    }

    /// Retourne les métriques d'utilisation du pool.
    ///
    /// # Retourne
    /// Un clone des métriques actuelles.
    pub fn metrics(&self) -> PoolMetrics {
        let inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.metrics.clone()
    }

    /// Vide complètement le pool de buffers.
    ///
    /// Tous les buffers stockés sont libérés et les métriques sont remises à zéro.
    pub fn clear(&self) {
        let mut inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.clear();
    }

    /// Retourne la mémoire totale utilisée par le pool (en octets).
    pub fn memory_usage(&self) -> usize {
        let inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.total_allocated
    }

    /// Retourne le nombre total de buffers stockés dans le pool.
    pub fn buffer_count(&self) -> usize {
        let inner = self
            .inner
            .lock()
            .expect("Verrou du pool de buffers corrompu");

        inner.pools.iter().map(|pool| pool.len()).sum()
    }
}

impl BufferPoolImpl {
    /// Acquiert un buffer de la taille minimale spécifiée.
    ///
    /// Cherche un buffer réutilisable dans les pools de la plus grande
    /// taille à la plus petite. Si aucun n'est compatible, alloue un
    /// nouveau buffer.
    fn acquire_buffer(&mut self, min_size: usize) -> Vec<u8> {
        // Calcul de l'index du bucket minimum requis
        let min_bucket = min_size / self.min_buffer_size;

        // Cherche un buffer compatible en partant du plus grand bucket
        for bucket in (min_bucket..self.num_buckets).rev() {
            if let Some(buffer) = self.pools[bucket].pop() {
                // Vérifie que le buffer est assez grand
                if buffer.capacity() >= min_size {
                    self.total_allocated -= buffer.capacity();

                    if self.enable_metrics {
                        self.metrics.record_reuse(buffer.capacity());
                        self.metrics.update_memory_usage(self.total_allocated);
                    }

                    // Retourne le buffer avec len=0, prêt à être rempli
                    let mut buffer = buffer;
                    buffer.clear(); // Reset len à 0 mais garde la capacité
                    return buffer;
                }
                // Buffer trop petit, remet dans le pool
                self.pools[bucket].push(buffer);
            }
        }

        // Aucun buffer réutilisable trouvé, allouer un nouveau
        // Arrondit la capacité au multiple supérieur de min_buffer_size
        let capacity = ((min_size / self.min_buffer_size) + 1) * self.min_buffer_size;

        if self.enable_metrics {
            self.metrics.record_new_allocation(capacity);
            self.metrics.record_pool_empty();
        }

        // Retourne un buffer avec len=0 et capacity>=min_size
        // Le buffer est prêt à recevoir des données via extend_from_slice
        Vec::with_capacity(capacity)
    }

    /// Remet un buffer dans le pool pour réutilisation.
    ///
    /// Si le pool a atteint sa limite mémoire, le buffer est ignoré
    /// (sera libéré par le Drop de Vec<u8>).
    fn release_buffer(&mut self, buffer: Vec<u8>) {
        let buffer_capacity = buffer.capacity();

        // Si le buffer est trop petit, on ne le stocke pas
        if buffer_capacity < self.min_buffer_size {
            return;
        }

        // Vérifie si on peut stocker ce buffer sans dépasser la limite
        if self.total_allocated + buffer_capacity > self.max_pool_memory {
            if self.enable_metrics {
                self.metrics.record_pool_full();
            }
            // Ne stocke pas le buffer, il sera libéré par Drop
            return;
        }

        // Calcul de l'index du bucket pour ce buffer
        let bucket = min(buffer_capacity / self.min_buffer_size, self.num_buckets - 1);

        self.pools[bucket].push(buffer);
        self.total_allocated += buffer_capacity;

        if self.enable_metrics {
            self.metrics.record_release(buffer_capacity);
            self.metrics.update_memory_usage(self.total_allocated);
        }
    }

    /// Vide complètement le pool et remet les métriques à zéro.
    fn clear(&mut self) {
        for pool in &mut self.pools {
            pool.clear();
        }
        self.total_allocated = 0;

        if self.enable_metrics {
            self.metrics.reset();
        }
    }
}
