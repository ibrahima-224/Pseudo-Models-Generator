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

//! # Tests du Module Pool de Buffers
//!
//! Tests unitaires pour le pool de buffers unifié, les métriques,
//! le pool typé et le buffer avec remise automatique.

#[cfg(test)]
mod tests {
    use super::super::typed_pool::TypedPool;
    use super::super::{PoolConfig, UnifiedBufferPool};

    /// Configuration de test avec petite taille min pour les buffers.
    fn test_config() -> PoolConfig {
        PoolConfig::new(32 * 1024 * 1024, 1024, 64 * 1024 * 1024, true)
    }

    /// Test de création et utilisation basique du pool avec PooledBuffer.
    #[test]
    fn test_pool_basic_acquire_release() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Acquérir un PooledBuffer u8 (retourne automatiquement au poolwhen dropped)
        let mut buf = pool.acquire_pooled_u8(1024);
        // Le buffer a len=0 et capacity>=1024, prêt à recevoir des données
        assert!(buf.capacity() >= 1024);

        // Écrire des données via extend_from_slice
        buf.extend_from_slice(&[42, 100]);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0], 42);
        assert_eq!(buf[1], 100);

        // Le buffer est automatiquement retourné au poolwhen dropped
        drop(buf);

        // Vérifie que le pool contient un buffer
        assert!(pool.buffer_count() > 0);
    }

    /// Test de la réutilisation des buffers avec PooledBuffer.
    #[test]
    fn test_pool_reuse_buffer() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Premier acquire - allocation neuve
        let buf1 = pool.acquire_pooled_u8(2048);
        assert!(buf1.capacity() >= 2048);
        assert_eq!(buf1.len(), 0); // len=0, prêt à être rempli
        drop(buf1);

        // Deuxième acquire - devrait réutiliser
        let buf2 = pool.acquire_pooled_u8(2048);
        assert!(buf2.capacity() >= 2048);
        assert_eq!(buf2.len(), 0); // len=0 après clear()
        drop(buf2);

        // Vérifie les métriques
        let metrics = pool.metrics();
        assert!(metrics.buffer_reuses > 0);
    }

    /// Test des métriques du pool.
    #[test]
    fn test_pool_metrics() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Acquérir et libérer plusieurs buffers
        for _ in 0..5 {
            let buf = pool.acquire_pooled_u8(2048);
            drop(buf);
        }

        let metrics = pool.metrics();
        assert!(metrics.total_acquisitions >= 5);
        assert!(metrics.buffers_released >= 4); // Au moins 4 retours
        assert!(metrics.reuse_rate() >= 0.0);
    }

    /// Test de la limite mémoire du pool.
    #[test]
    fn test_pool_memory_limit() {
        // Pool avec limite très petite (4 Ko) pour forcer le débordement
        let config = PoolConfig::new(4096, 1024, 64 * 1024 * 1024, true);
        let pool = UnifiedBufferPool::new(config);

        // Acquérir et libérer de grands buffers
        for _ in 0..10 {
            let buf = pool.acquire_pooled_u8(2048);
            drop(buf);
        }

        // La mémoire ne devrait pas dépasser la limite (4 Ko)
        assert!(pool.memory_usage() <= 4096);
    }

    /// Test du pool typé pour u8.
    #[test]
    fn test_typed_pool_u8() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);
        let typed_pool = TypedPool::<u8>::new_u8(pool);

        let mut buf = typed_pool.acquire(2048);
        // Le buffer a len=0 et capacity>=2048
        assert!(buf.capacity() >= 2048);

        // Écrire des données via extend_from_slice
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        buf.extend_from_slice(&data);
        assert_eq!(buf.len(), 2048);

        drop(buf);
    }

    /// Test du pool typé pour f64.
    #[test]
    fn test_typed_pool_f64() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);
        let typed_pool = TypedPool::<f64>::new_f64(pool);

        let mut buf = typed_pool.acquire(128);
        // Le buffer a len=0 et capacity>=128 éléments f64
        assert!(buf.capacity() >= 128);

        // Écrire des données via extend_from_slice
        let data: Vec<f64> = (0..128).map(|i| i as f64 * 0.1).collect();
        buf.extend_from_slice(&data);
        assert_eq!(buf.len(), 128);

        // Vérifier les données
        for i in 0..128 {
            assert!(
                (buf[i] - i as f64 * 0.1).abs() < 1e-10,
                "Échec à l'index {}: obtenu {}, attendu {}",
                i,
                buf[i],
                i as f64 * 0.1
            );
        }

        drop(buf);
    }

    /// Test du buffer avec remise automatique (PooledBuffer).
    #[test]
    fn test_pooled_buffer_drop() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);
        let typed_pool = TypedPool::<u8>::new_u8(pool.clone());

        {
            let _buf = typed_pool.acquire(2048);
            // Buffer est actif dans cette portée
            assert!(pool.buffer_count() == 0);
        }
        // Buffer devrait être retourné au pool ici

        assert!(pool.buffer_count() > 0);
    }

    /// Test du Deref/DerefMut sur PooledBuffer.
    #[test]
    fn test_pooled_buffer_deref() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);
        let typed_pool = TypedPool::<u8>::new_u8(pool);

        let mut buf = typed_pool.acquire(2048);

        // Écrire des données via extend_from_slice
        buf.extend_from_slice(&[42, 100, 200]);
        assert_eq!(buf.len(), 3);

        // Accès via Deref
        assert_eq!(buf[0], 42);
        assert_eq!(buf[1], 100);
        assert_eq!(buf[2], 200);
    }

    /// Test de thread-safety du pool.
    #[test]
    fn test_pool_thread_safety() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let pool_clone = pool.clone();
                std::thread::spawn(move || {
                    for j in 0..10 {
                        // Réduit de 100 à 10 itérations pour optimisation mémoire
                        let mut buf = pool_clone.acquire_pooled_u8(1024 + i * 256 + j);
                        // Écrire via extend_from_slice
                        buf.extend_from_slice(&[i as u8]);
                        drop(buf);
                    }
                })
            })
            .collect();

        // Attendre la fin de tous les threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Vérifie que le pool est dans un état cohérent
        assert!(pool.buffer_count() <= 400);
        assert!(pool.memory_usage() <= 32 * 1024 * 1024);
    }

    /// Test de la configuration du pool.
    #[test]
    fn test_pool_config() {
        let config = PoolConfig::default();
        assert_eq!(config.max_memory_per_pool, 32 * 1024 * 1024);
        assert_eq!(config.min_buffer_size, 1024 * 1024);
        assert_eq!(config.max_buffer_size, 64 * 1024 * 1024);
        assert!(config.enable_metrics);

        // Test des configurations prédéfinies
        let small = PoolConfig::small_chunks();
        assert_eq!(small.max_buffer_size, 8 * 1024 * 1024);

        let large = PoolConfig::large_chunks();
        assert_eq!(large.max_buffer_size, 64 * 1024 * 1024);
    }

    /// Test de validation de la configuration.
    #[test]
    fn test_pool_config_validation() {
        // Configuration invalide : min > max
        let invalid = PoolConfig::new(1024, 2048, 1024, true);
        assert!(invalid.validate().is_err());

        // Configuration invalide : mémoire = 0
        let invalid_zero = PoolConfig::new(0, 1024, 2048, true);
        assert!(invalid_zero.validate().is_err());

        // Configuration valide
        let valid = PoolConfig::new(1024, 512, 2048, true);
        assert!(valid.validate().is_ok());
    }

    /// Test du reset des métriques via clear().
    #[test]
    fn test_pool_metrics_reset() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Générer quelques métriques
        let _buf = pool.acquire_pooled_u8(2048);

        let metrics_before = pool.metrics();
        assert!(metrics_before.total_acquisitions > 0);

        // Le reset est testé implicitement via clear()
        pool.clear();
        let metrics_after = pool.metrics();
        assert_eq!(metrics_after.total_acquisitions, 0);
    }

    /// Test de la méthode into_inner sur PooledBuffer.
    #[test]
    fn test_pooled_buffer_into_inner() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);
        let typed_pool = TypedPool::<u8>::new_u8(pool.clone());

        let mut buf = typed_pool.acquire(2048);
        let initial_count = pool.buffer_count();

        // Écrire des données pour avoir len > 0
        buf.extend_from_slice(&[1, 2, 3, 4, 5]);

        // Extraire le buffer sans le retourner au pool
        let inner = buf.into_inner();
        assert_eq!(inner.len(), 5);

        // Le buffer n'est pas retourné au pool car on l'a extrait
        assert_eq!(pool.buffer_count(), initial_count);

        // L'buffer extrait est géré manuellement
        drop(inner);
    }

    /// Test de l'acquisition directe avec release_u8.
    #[test]
    fn test_acquire_release_u8_manual() {
        let config = test_config();
        let pool = UnifiedBufferPool::new(config);

        // Acquérir un buffer brut
        let buf = pool.acquire_u8(2048);
        // Le buffer a len=0 et capacity>=2048
        assert!(buf.capacity() >= 2048);

        // Libérer manuellement le buffer
        pool.release_u8(buf);

        // Vérifie que le pool contient un buffer
        assert!(pool.buffer_count() > 0);
    }
}
