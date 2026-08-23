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

//! # Métriques du Pool de Buffers
//!
//! Ce module fournit des métriques centralisées pour surveiller
//! l'utilisation du pool de buffers unifié.

use std::fmt;

/// Métriques centralisées d'utilisation du pool de buffers.
///
/// Ces métriques permettent de suivre les performances du pool :
/// - Nombre total d'acquisitions de buffers
/// - Nombre de réutilisations de buffers (hit rate)
/// - Mémoire totale utilisée et pic
/// - Nombre de buffers créés et libérés
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    /// Nombre total d'acquisitions de buffers (nouveaux + réutilisés).
    pub total_acquisitions: u64,

    /// Nombre de réutilisations de buffers (buffers retournés du pool).
    pub buffer_reuses: u64,

    /// Nombre de nouveaux buffers alloués (pas trouvés dans le pool).
    pub new_allocations: u64,

    /// Nombre de buffers retournés au pool.
    pub buffers_released: u64,

    /// Mémoire totale actuellement utilisée par les buffers actifs (en octets).
    pub current_memory_usage: usize,

    /// Mémoire maximale atteinte simultanément (en octets).
    pub peak_memory_usage: usize,

    /// Nombre de fois où le pool était vide (aucun buffer disponible).
    pub pool_empty_count: u64,

    /// Nombre de fois où un buffer a été refusé (mémoire pool dépassée).
    pub pool_full_count: u64,

    /// Nombre total d'octets alloués pour les nouveaux buffers.
    pub total_bytes_allocated: u64,

    /// Nombre total d'octets retournés au pool.
    pub total_bytes_released: u64,
}

impl PoolMetrics {
    /// Crée des métriques vides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre une acquisition de buffer (réutilisé du pool).
    pub fn record_reuse(&mut self, buffer_size: usize) {
        self.total_acquisitions += 1;
        self.buffer_reuses += 1;
        self.total_bytes_released += buffer_size as u64;
    }

    /// Enregistre une nouvelle allocation de buffer.
    pub fn record_new_allocation(&mut self, buffer_size: usize) {
        self.total_acquisitions += 1;
        self.new_allocations += 1;
        self.total_bytes_allocated += buffer_size as u64;
    }

    /// Enregistre le retour d'un buffer au pool.
    pub fn record_release(&mut self, buffer_size: usize) {
        self.buffers_released += 1;
        self.total_bytes_released += buffer_size as u64;
    }

    /// Met à jour l'utilisation mémoire actuelle.
    pub fn update_memory_usage(&mut self, current: usize) {
        self.current_memory_usage = current;
        if current > self.peak_memory_usage {
            self.peak_memory_usage = current;
        }
    }

    /// Enregistre un événement de pool vide.
    pub fn record_pool_empty(&mut self) {
        self.pool_empty_count += 1;
    }

    /// Enregistre un événement de pool plein.
    pub fn record_pool_full(&mut self) {
        self.pool_full_count += 1;
    }

    /// Calcule le taux de réutilisation des buffers (hit rate).
    ///
    /// # Retourne
    /// Le pourcentage de buffers réutilisés par rapport au total des acquisitions.
    /// Retourne 0.0 si aucune acquisition n'a été effectuée.
    pub fn reuse_rate(&self) -> f64 {
        if self.total_acquisitions == 0 {
            return 0.0;
        }
        (self.buffer_reuses as f64 / self.total_acquisitions as f64) * 100.0
    }

    /// Calcule le nombre moyen de réutilisations par buffer.
    pub fn avg_reuses_per_buffer(&self) -> f64 {
        if self.buffers_released == 0 {
            return 0.0;
        }
        self.buffer_reuses as f64 / self.buffers_released as f64
    }

    /// Remet toutes les métriques à zéro.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for PoolMetrics {
    /// Affiche les métriques du pool de manière lisible.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Métriques du Pool de Buffers ===")?;
        writeln!(f, "  Acquisitions totales : {}", self.total_acquisitions)?;
        writeln!(f, "  Réutilisations : {}", self.buffer_reuses)?;
        writeln!(f, "  Nouvelles allocations : {}", self.new_allocations)?;
        writeln!(f, "  Buffers retournés : {}", self.buffers_released)?;
        writeln!(f, "  Taux de réutilisation : {:.1}%", self.reuse_rate())?;
        writeln!(
            f,
            "  Mémoire actuelle : {} octets",
            self.current_memory_usage
        )?;
        writeln!(f, "  Mémoire pic : {} octets", self.peak_memory_usage)?;
        writeln!(f, "  Pool vide : {} fois", self.pool_empty_count)?;
        writeln!(f, "  Pool plein : {} fois", self.pool_full_count)?;
        writeln!(f, "  Octets alloués : {}", self.total_bytes_allocated)?;
        writeln!(f, "  Octets libérés : {}", self.total_bytes_released)?;
        Ok(())
    }
}
