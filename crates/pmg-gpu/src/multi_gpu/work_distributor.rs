//! Distributeur de travail multi-GPU
//!
//! Ce module fournit la structure `WorkDistributor` pour distribuer
//! efficacement les charges de travail sur plusieurs devices GPU.
//! Il gère la parallélisation, le suivi des statistiques et
//! l'équilibrage de charge entre les devices.

use std::sync::{Arc, Mutex};
use std::thread;

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::multi_gpu::MultiGpuPool;

/// Distributeur de travail multi-GPU
///
/// Gère la distribution parallèle de tâches sur plusieurs devices GPU.
#[derive(Debug)]
pub struct WorkDistributor {
    /// Pool de devices GPU
    pool: Arc<MultiGpuPool>,
    /// Taille du chunk par device (nombre d'éléments par tâche)
    chunk_size: usize,
    /// Nombre maximum de threads parallèles
    max_threads: usize,
}

impl WorkDistributor {
    /// Crée un nouveau distributeur de travail
    ///
    /// # Arguments
    ///
    /// * `pool` - Pool de devices GPU à utiliser
    /// * `chunk_size` - Nombre d'éléments à traiter par device
    ///
    /// # Exemple
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use pmg_gpu::multi_gpu::{MultiGpuPool, WorkDistributor};
    ///
    /// let pool = Arc::new(MultiGpuPool::new().unwrap());
    /// let distributor = WorkDistributor::new(pool, 1024);
    /// ```
    pub fn new(pool: Arc<MultiGpuPool>, chunk_size: usize) -> Self {
        // Utiliser std::thread::available_parallelism() au lieu de num_cpus
        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(2); // Au moins 2 threads

        Self {
            pool,
            chunk_size,
            max_threads,
        }
    }

    /// Crée un distributeur avec un nombre maximum de threads spécifié
    pub fn with_max_threads(
        pool: Arc<MultiGpuPool>,
        chunk_size: usize,
        max_threads: usize,
    ) -> Self {
        Self {
            pool,
            chunk_size,
            max_threads: max_threads.max(1),
        }
    }

    /// Distribue un travail sur plusieurs GPU
    ///
    /// # Arguments
    ///
    /// * `work_items` - Éléments à traiter
    /// * `worker` - Fonction de traitement pour chaque élément (wrapped dans Arc)
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la distribution échoue.
    pub fn distribute<T, F, R>(&self, work_items: Vec<T>, worker: Arc<F>) -> GpuResult<Vec<Vec<R>>>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
        F: Fn(T, Arc<Mutex<GpuDevice>>) -> GpuResult<R> + Send + Sync + 'static,
    {
        if work_items.is_empty() {
            return Ok(Vec::new());
        }

        if self.pool.device_count() == 0 {
            return Err(GpuError::MultiGpuError(
                "Aucun device disponible".to_string(),
            ));
        }

        // Transformer work_items en Arc pour partager les données entre les threads sans copie
        let shared_items = Arc::new(work_items);
        let total_items = shared_items.len();

        // Calculer les indices de début et de fin pour chaque chunk
        let mut chunk_ranges = Vec::new();
        let mut start = 0;
        while start < total_items {
            let end = (start + self.chunk_size).min(total_items);
            chunk_ranges.push((start, end));
            start = end;
        }

        let mut handles = Vec::with_capacity(chunk_ranges.len());

        // Distribuer les chunks sur les devices
        for (start_idx, end_idx) in chunk_ranges {
            let pool = self.pool.clone();
            let worker = worker.clone();
            let items_arc = shared_items.clone();

            let handle = thread::spawn(move || {
                // Sélectionner un device selon la stratégie
                let device = pool.next_device()?;
                let device_id = {
                    let device_guard = device.lock().unwrap();
                    device_guard.info().id
                };

                let start_time = std::time::Instant::now();
                let mut results = Vec::with_capacity(end_idx - start_idx);

                // Traiter chaque élément du chunk en clonant uniquement les éléments nécessaires
                for idx in start_idx..end_idx {
                    // Cloner l'élément individuellement pour éviter la copie du chunk entier
                    let item = items_arc[idx].clone();
                    match worker(item, device.clone()) {
                        Ok(result) => results.push(result),
                        Err(e) => {
                            log::error!("Erreur traitement sur device {}: {}", device_id, e);
                            // Mettre à jour les statistiques avant de retourner l'erreur
                            let elapsed = start_time.elapsed().as_millis() as u64;
                            pool.update_stats(device_id, elapsed, 0, true);
                            return Err(e);
                        },
                    }
                }

                // Mettre à jour les statistiques en cas de succès
                let elapsed = start_time.elapsed().as_millis() as u64;
                pool.update_stats(device_id, elapsed, 0, false);

                Ok(results)
            });

            handles.push(handle);

            // Limiter le nombre de threads parallèles
            if handles.len() >= self.max_threads {
                break;
            }
        }

        // Attendre tous les résultats
        let mut all_results = Vec::new();

        for handle in handles {
            match handle.join() {
                Ok(Ok(results)) => all_results.push(results),
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(GpuError::MultiGpuError("Thread panic".to_string())),
            }
        }

        Ok(all_results)
    }

    /// Distribue un travail avec répartition automatique
    ///
    /// Cette méthode répartit automatiquement le travail entre tous les devices
    /// disponibles pour une utilisation optimale.
    pub fn distribute_auto<T, F, R>(
        &self,
        work_items: Vec<T>,
        worker: Arc<F>,
    ) -> GpuResult<Vec<Vec<R>>>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
        F: Fn(T, Arc<Mutex<GpuDevice>>) -> GpuResult<R> + Send + Sync + 'static,
    {
        let device_count = self.pool.device_count();

        if device_count == 0 {
            return Err(GpuError::MultiGpuError(
                "Aucun device disponible".to_string(),
            ));
        }

        // Calculer la taille optimale des chunks
        let optimal_chunk_size = (work_items.len() / device_count).max(1);
        let distributor =
            Self::with_max_threads(self.pool.clone(), optimal_chunk_size, self.max_threads);

        distributor.distribute(work_items, worker)
    }

    /// Retourne le pool de devices utilisé
    pub fn pool(&self) -> &Arc<MultiGpuPool> {
        &self.pool
    }

    /// Retourne la taille des chunks
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Retourne le nombre maximum de threads
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }
}

/// Helper pour créer un pool et un distributeur facilement
pub fn create_multi_gpu_system(
    strategy: crate::multi_gpu::DeviceStrategy,
    chunk_size: usize,
) -> GpuResult<(Arc<MultiGpuPool>, WorkDistributor)> {
    let pool = Arc::new(MultiGpuPool::with_strategy(strategy)?);
    let distributor = WorkDistributor::new(pool.clone(), chunk_size);
    Ok((pool, distributor))
}

/// Alias pour compatibilité avec lib.rs
pub type MultiGpuManager = MultiGpuPool;
