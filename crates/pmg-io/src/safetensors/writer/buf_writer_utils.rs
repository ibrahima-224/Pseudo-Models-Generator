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

//! Utilitaires pour le writer SafeTensors optimisé.
//!
//! Ce module contient les structures de support pour le pool de buffers
//! et les métriques de performance du writer optimisé.

use std::cmp::min;

/// Taille minimale du buffer (1 Mo).
#[allow(dead_code)]
pub const MIN_BUFFER_SIZE: usize = 1024 * 1024;

/// Taille maximale du buffer (64 Mo).
#[allow(dead_code)]
pub const MAX_BUFFER_SIZE: usize = 64 * 1024 * 1024;

/// Pool de buffers réutilisables pour optimiser la mémoire.
///
/// Ce pool permet de réutiliser les buffers d'écriture pour éviter
/// les allocations successives et réduire la fragmentation mémoire.
#[allow(dead_code)]
pub struct BufferPool {
    /// Buffers disponibles par taille (index = taille / MIN_BUFFER_SIZE).
    pools: Vec<Vec<Vec<u8>>>,
    /// Mémoire totale allouée dans le pool.
    total_allocated: usize,
    /// Mémoire maximale du pool (32 Mo par défaut).
    max_pool_memory: usize,
}

#[allow(dead_code)]
impl BufferPool {
    /// Crée un nouveau pool de buffers avec une mémoire maximale spécifiée.
    ///
    /// # Paramètres
    /// - `max_memory` : mémoire maximale autorisée pour le pool (en octets)
    ///
    /// # Retourne
    /// Un pool de buffers prêt à l'emploi.
    pub fn new(max_memory: usize) -> Self {
        let num_pools = MAX_BUFFER_SIZE / MIN_BUFFER_SIZE + 1;
        let pools = vec![Vec::new(); num_pools];

        Self {
            pools,
            total_allocated: 0,
            max_pool_memory: max_memory,
        }
    }

    /// Obtient un buffer de la taille spécifiée (réutilise si possible).
    ///
    /// # Paramètres
    /// - `min_size` : taille minimale requise pour le buffer
    ///
    /// # Retourne
    /// Un buffer de taille au moins `min_size`.
    pub fn acquire(&mut self, min_size: usize) -> Vec<u8> {
        let pool_index = min(min_size / MIN_BUFFER_SIZE, self.pools.len() - 1);

        // Chercher un buffer compatible dans les pools de taille >= pool_index
        for idx in pool_index..self.pools.len() {
            if let Some(buffer) = self.pools[idx].pop() {
                self.total_allocated -= buffer.capacity();
                return buffer;
            }
        }

        // Aucun buffer réutilisable trouvé, allouer un nouveau buffer
        let capacity = (pool_index + 1) * MIN_BUFFER_SIZE;
        Vec::with_capacity(capacity)
    }

    /// Remet un buffer dans le pool pour réutilisation.
    ///
    /// # Paramètres
    /// - `buffer` : buffer à remettre dans le pool
    pub fn release(&mut self, buffer: Vec<u8>) {
        let buffer_capacity = buffer.capacity();
        if self.total_allocated + buffer_capacity > self.max_pool_memory {
            return;
        }

        let pool_index = min(buffer_capacity / MIN_BUFFER_SIZE, self.pools.len() - 1);
        self.pools[pool_index].push(buffer);
        self.total_allocated += buffer_capacity;
    }

    /// Retourne la mémoire totale utilisée par le pool.
    ///
    /// # Retourne
    /// La mémoire totale allouée dans le pool (en octets).
    pub fn memory_usage(&self) -> usize {
        self.total_allocated
    }

    /// Retourne le nombre total de buffers stockés dans le pool.
    ///
    /// # Retourne
    /// Le nombre total de buffers disponibles.
    pub fn buffer_count(&self) -> usize {
        self.pools.iter().map(|pool| pool.len()).sum()
    }

    /// Vide complètement le pool de buffers.
    pub fn clear(&mut self) {
        for pool in &mut self.pools {
            pool.clear();
        }
        self.total_allocated = 0;
    }
}

/// Métriques de performance de l'écriture pour BufWriter.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct BufWriterMetrics {
    /// Nombre total d'octets écrits.
    pub bytes_written: u64,
    /// Nombre de tenseurs écrits.
    pub tensors_written: usize,
    /// Temps total d'écriture en secondes.
    pub write_time_secs: f64,
    /// Vitesse moyenne d'écriture en Mo/s.
    pub avg_speed_mbps: f64,
    /// Nombre d'appels système write effectués.
    pub write_syscalls: usize,
    /// Mémoire maximale utilisée simultanément.
    pub peak_memory_usage: usize,
    /// Nombre de réutilisations de buffers.
    pub buffer_reuses: usize,
}

#[allow(dead_code)]
impl BufWriterMetrics {
    /// Crée des métriques vides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Met à jour les métriques avec une nouvelle écriture.
    pub fn update_write(&mut self, bytes: usize, time_secs: f64) {
        self.bytes_written += bytes as u64;
        self.tensors_written += 1;
        self.write_time_secs += time_secs;
        self.write_syscalls += 1;

        if self.write_time_secs > 0.0 {
            self.avg_speed_mbps =
                self.bytes_written as f64 / self.write_time_secs / 1024.0 / 1024.0;
        }
    }

    /// Incrémente le compteur de réutilisations de buffers.
    pub fn increment_buffer_reuse(&mut self) {
        self.buffer_reuses += 1;
    }

    /// Met à jour la mémoire maximale si nécessaire.
    pub fn update_peak_memory(&mut self, current_memory: usize) {
        if current_memory > self.peak_memory_usage {
            self.peak_memory_usage = current_memory;
        }
    }
}
