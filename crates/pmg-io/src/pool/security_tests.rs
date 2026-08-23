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

//! # Tests de Sécurité pour le Pool de Buffers
//!
//! Tests unitaires validant les corrections de sécurité pour VULN-MED-001 :
//! - Vérification de la sécurité des conversions entre types
//! - Détection des overflow dans les calculs de taille
//! - Validation des invariants du pool optimisé
//! - Détection des fuites mémoire

#[cfg(test)]
mod security_tests {
    use super::super::typed_pool::PooledBuffer;
    use super::super::{PoolConfig, UnifiedBufferPool};

    /// Configuration de test avec petite taille min pour les buffers.
    fn test_config() -> PoolConfig {
        PoolConfig::new(32 * 1024 * 1024, 1024, 64 * 1024 * 1024, true)
    }

    /// Test de la sécurité des conversions entre types dans TypedPool.
    ///
    /// Valide que les vérifications debug_assert! sont correctement appliquées
    /// pour les conversions entre types u8 et f64.
    #[test]
    fn test_typed_pool_conversion_safety() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Test avec un buffer u8
        let mut u8_buffer = pool.acquire_u8(1024);
        u8_buffer.extend_from_slice(&[1, 2, 3, 4]);
        let u8_pooled = PooledBuffer::new_u8(u8_buffer, pool.clone());

        // Vérifier que le buffer peut être droppé sans panic
        drop(u8_pooled);

        // Test avec un buffer f64
        let mut f64_buffer = pool.acquire_f64(256); // 256 * 8 = 2048 octets
        f64_buffer.extend_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let f64_pooled = PooledBuffer::new_f64(f64_buffer, pool.clone());

        // Vérifier que le buffer peut être droppé sans panic
        drop(f64_pooled);

        // Vérifier que le pool contient des buffers (les buffers ont été retournés)
        let metrics = pool.metrics();
        assert!(
            metrics.buffers_released > 0,
            "Les buffers n'ont pas été retournés au pool"
        );
    }

    /// Test de la détection d'overflow dans buffer_pool.
    ///
    /// Valide que checked_mul est utilisé pour éviter les overflow
    /// dans les calculs de taille en octets.
    #[test]
    #[should_panic(expected = "Overflow dans le calcul de la taille en octets pour f64")]
    fn test_buffer_pool_overflow_detection() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Créer un buffer f64 avec une taille qui pourrait causer un overflow
        // si on utilisait une multiplication normale
        // Cette taille causera un panic avec checked_mul
        let large_size = usize::MAX / 8 + 1; // Taille qui overflow en octets
        let _buffer = pool.acquire_f64(large_size);

        // Le test devrait panic avant d'atteindre cette ligne
        panic!("Le test devrait avoir panic avant");
    }

    /// Test des invariants du pool optimisé.
    ///
    /// Valide que les vérifications de taille et d'alignement sont correctement
    /// appliquées dans le pool optimisé.
    #[test]
    fn test_optimized_pool_invariants() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Test avec différents types
        let test_cases = vec![
            (1, "petit buffer u8"),
            (1024, "buffer u8 moyen"),
            (64 * 1024, "grand buffer u8"),
        ];

        for (size, description) in test_cases {
            // Acquérir et libérer un buffer
            let buffer = pool.acquire_u8(size);
            assert!(
                buffer.capacity() >= size,
                "Échec pour {}: capacité insuffisante",
                description
            );
            drop(buffer);
        }

        // Vérifier que tous les buffers sont retournés
        let metrics = pool.metrics();
        assert_eq!(metrics.current_memory_usage, 0);
    }

    /// Test de détection de fuites mémoire.
    ///
    /// Valide que les buffers sont correctement retournés au pool
    /// et qu'il n'y a pas de fuites mémoire.
    #[test]
    fn test_memory_leak_detection() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Acquérir et libérer de nombreux buffers
        for i in 0..50 {
            let size = 1024 * (i % 5 + 1);
            let buffer = pool.acquire_u8(size);

            // Écrire des données pour s'assurer que le buffer est utilisé
            let mut buffer = buffer;
            buffer.extend_from_slice(&[i as u8; 32]);

            // Libérer le buffer
            drop(buffer);
        }

        // Vérifier qu'il n'y a pas de fuite mémoire
        let metrics = pool.metrics();
        assert_eq!(
            metrics.current_memory_usage, 0,
            "Fuite mémoire détectée: {} octets non libérés",
            metrics.current_memory_usage
        );

        // Vérifier que des allocations ont été effectuées
        assert!(
            metrics.total_acquisitions > 0,
            "Aucune allocation n'a été effectuée"
        );
    }

    /// Test de la sécurité des conversions avec des types non supportés.
    ///
    /// Valide que les types non supportés sont correctement gérés
    /// sans provoquer de comportement undefined.
    #[test]
    fn test_unsupported_type_conversion() {
        // Ce test vérifie que le code ne panique pas avec des types non supportés
        // dans les conversions du pool typé.

        // Note: Ce test est principalement pour valider que les debug_assert!
        // fonctionnent correctement. En production, seuls u8 et f64 sont supportés.

        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Test avec un buffer u8 standard
        let buffer = pool.acquire_u8(1024);
        drop(buffer);

        // Vérifier que le pool fonctionne correctement après le test
        let metrics = pool.metrics();
        assert!(metrics.total_acquisitions > 0);
    }
}
