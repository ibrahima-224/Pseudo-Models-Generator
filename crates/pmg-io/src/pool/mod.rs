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

//! # Module Pool de Buffers Unifié
//!
//! Ce module fournit un pool de buffers unifié et thread-safe pour optimiser
//! les allocations mémoire dans le pipeline de génération.
//!
//! ## Composants
//!
//! - [`UnifiedBufferPool`] : Pool principal thread-safe avec gestion mémoire
//! - [`TypedPool`] : Pool typé pour les types spécifiques (u8, f64)
//! - [`PooledBuffer`] : Buffer avec remise automatique dans le pool (Drop)
//! - [`PoolMetrics`] : Métriques centralisées d'utilisation du pool
//!
//! ## Utilisation
//!
//! ```rust,ignore
//! use pmg_io::pool::{UnifiedBufferPool, PoolConfig};
//!
//! let config = PoolConfig::default();
//! let pool = UnifiedBufferPool::new(config);
//!
//! // Acquérir un buffer u8
//! let mut buffer = pool.acquire_u8(1024);
//! buffer.extend_from_slice(b"données");
//! // Le buffer est automatiquement remis dans le poolwhen dropped
//! ```

pub mod buffer_pool;
pub mod metrics;
pub mod optimized;
#[cfg(test)]
mod security_tests;
#[cfg(test)]
mod tests;
pub mod typed_pool;

// Réexports publics pour faciliter l'usage
pub use buffer_pool::UnifiedBufferPool;
pub use metrics::PoolMetrics;
pub use optimized::{OptimizedBufferPool, PoolStats};
pub use typed_pool::{PooledBuffer, TypedPool};

/// Configuration du pool de buffers.
///
/// Cette structure permet de configurer les paramètres du pool de buffers
/// unifié, notamment les limites mémoire et les tailles de buffers.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Mémoire maximale par pool (en octets).
    /// Défaut : 32 Mo (32 * 1024 * 1024)
    pub max_memory_per_pool: usize,

    /// Taille minimale des buffers (en octets).
    /// Défaut : 1 Mo (1024 * 1024)
    pub min_buffer_size: usize,

    /// Taille maximale des buffers (en octets).
    /// Défaut : 64 Mo (64 * 1024 * 1024)
    pub max_buffer_size: usize,

    /// Activer les métriques détaillées d'utilisation du pool.
    /// Défaut : true
    pub enable_metrics: bool,
}

impl Default for PoolConfig {
    /// Crée une configuration par défaut du pool de buffers.
    ///
    /// - Mémoire maximale : 32 Mo par pool
    /// - Taille minimale buffer : 1 Mo
    /// - Taille maximale buffer : 64 Mo
    /// - Métriques activées
    fn default() -> Self {
        Self {
            max_memory_per_pool: 32 * 1024 * 1024, // 32 Mo
            min_buffer_size: 1024 * 1024,          // 1 Mo
            max_buffer_size: 64 * 1024 * 1024,     // 64 Mo
            enable_metrics: true,
        }
    }
}

impl PoolConfig {
    /// Crée une nouvelle configuration avec les paramètres spécifiés.
    ///
    /// # Paramètres
    /// - `max_memory_per_pool` : mémoire maximale par pool (en octets)
    /// - `min_buffer_size` : taille minimale des buffers (en octets)
    /// - `max_buffer_size` : taille maximale des buffers (en octets)
    /// - `enable_metrics` : activer les métriques détaillées
    pub fn new(
        max_memory_per_pool: usize,
        min_buffer_size: usize,
        max_buffer_size: usize,
        enable_metrics: bool,
    ) -> Self {
        Self {
            max_memory_per_pool,
            min_buffer_size,
            max_buffer_size,
            enable_metrics,
        }
    }

    /// Crée une configuration optimisée pour les petits chunks (1-8 Mo).
    ///
    /// Utile pour le streaming avec des chunks de taille réduite.
    pub fn small_chunks() -> Self {
        Self {
            max_memory_per_pool: 16 * 1024 * 1024, // 16 Mo
            min_buffer_size: 512 * 1024,           // 512 Ko
            max_buffer_size: 8 * 1024 * 1024,      // 8 Mo
            enable_metrics: true,
        }
    }

    /// Crée une configuration optimisée pour les gros chunks (8-64 Mo).
    ///
    /// Utile pour la génération de modèles de grande taille.
    pub fn large_chunks() -> Self {
        Self {
            max_memory_per_pool: 128 * 1024 * 1024, // 128 Mo
            min_buffer_size: 4 * 1024 * 1024,       // 4 Mo
            max_buffer_size: 64 * 1024 * 1024,      // 64 Mo
            enable_metrics: true,
        }
    }

    /// Valide la cohérence de la configuration.
    ///
    /// # Erreurs
    /// Retourne une erreur si la configuration est invalide :
    /// - `min_buffer_size` > `max_buffer_size`
    /// - `max_memory_per_pool` == 0
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_buffer_size > self.max_buffer_size {
            return Err("min_buffer_size doit être <= max_buffer_size");
        }
        if self.max_memory_per_pool == 0 {
            return Err("max_memory_per_pool doit être > 0");
        }
        Ok(())
    }
}
