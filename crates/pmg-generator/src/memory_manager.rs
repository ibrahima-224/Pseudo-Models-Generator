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

//! Gestionnaire de mémoire bornée pour l'itérateur streaming.
//!
//! Ce module fournit [`BoundedMemoryManager`] qui gère l'allocation de mémoire
//! pour l'itérateur, en s'assurant que la mémoire utilisée reste dans les limites
//! spécifiées. Cela est particulièrement utile pour les modèles de grande taille
//! où la consommation mémoire doit être contrôlée.

/// Mémoire bornée avec allocation progressive.
///
/// Cette structure gère l'allocation de mémoire pour l'itérateur,
/// en s'assurant que la mémoire utilisée reste dans les limites spécifiées.
pub struct BoundedMemoryManager {
    /// Mémoire maximale allouable (en octets).
    max_memory: u64,
    /// Mémoire actuellement utilisée.
    current_usage: u64,
    /// Taille des chunks pour l'allocation progressive.
    chunk_size: usize,
}

impl BoundedMemoryManager {
    /// Crée un nouveau gestionnaire de mémoire bornée.
    ///
    /// # Paramètres
    /// - `max_memory` : mémoire maximale en octets
    /// - `chunk_size` : taille des chunks pour l'allocation progressive
    pub fn new(max_memory: u64, chunk_size: usize) -> Self {
        Self {
            max_memory,
            current_usage: 0,
            chunk_size,
        }
    }

    /// Vérifie si l'allocation est possible.
    pub fn can_allocate(&self, size: u64) -> bool {
        self.current_usage + size <= self.max_memory
    }

    /// Alloue de la mémoire si possible.
    pub fn allocate(&mut self, size: u64) -> bool {
        if self.can_allocate(size) {
            self.current_usage += size;
            true
        } else {
            false
        }
    }

    /// Libère de la mémoire.
    pub fn deallocate(&mut self, size: u64) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    /// Retourne la mémoire utilisée.
    pub fn current_usage(&self) -> u64 {
        self.current_usage
    }

    /// Retourne la mémoire maximale.
    pub fn max_memory(&self) -> u64 {
        self.max_memory
    }

    /// Retourne la taille des chunks.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_memory_manager() {
        // Test avec une limite de 1024 octets pour plus de simplicité
        let mut manager = BoundedMemoryManager::new(1024, 128);

        // Test d'allocation simple
        assert!(manager.can_allocate(512));
        assert!(manager.allocate(512));
        assert_eq!(manager.current_usage(), 512);

        // Test d'allocation qui reste dans la limite
        assert!(manager.can_allocate(400)); // 512 + 400 = 912 < 1024
        assert!(manager.allocate(400));
        assert_eq!(manager.current_usage(), 912);

        // Test d'allocation qui dépasse la limite
        assert!(!manager.can_allocate(200)); // 912 + 200 = 1112 > 1024
        assert!(!manager.allocate(200));

        // Test de désallocation
        manager.deallocate(912);
        assert_eq!(manager.current_usage(), 0);
        assert!(manager.can_allocate(1024));
        assert!(manager.allocate(1024));
        assert_eq!(manager.current_usage(), 1024);

        // Plus de place disponible
        assert!(!manager.can_allocate(1));
    }
}
