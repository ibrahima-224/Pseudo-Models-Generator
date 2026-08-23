//! Pool de buffers réutilisables pour réduire les allocations
//!
//! Ce module fournit un pool de buffers pour éviter les allocations répétées
//! lors de la compression/décompression en streaming.

use std::collections::VecDeque;

/// Pool de buffers réutilisables
pub struct BufferPool {
    /// Files de buffers libres par taille
    buffers: VecDeque<Vec<u8>>,
    /// Capacité maximale du pool
    max_size: usize,
    /// Taille par défaut des buffers
    default_capacity: usize,
}

impl BufferPool {
    /// Crée un nouveau pool de buffers
    ///
    /// # Arguments
    /// * `max_size` - Nombre maximum de buffers à maintenir dans le pool
    /// * `default_capacity` - Capacité par défaut des nouveaux buffers en octets
    pub fn new(max_size: usize, default_capacity: usize) -> Self {
        Self {
            buffers: VecDeque::with_capacity(max_size),
            max_size,
            default_capacity,
        }
    }

    /// Récupère un buffer du pool ou en crée un nouveau
    ///
    /// Si des buffers sont disponibles dans le pool, retourne le premier
    /// sinon crée un nouveau buffer avec la capacité par défaut.
    pub fn get(&mut self) -> Vec<u8> {
        self.buffers
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(self.default_capacity))
    }

    /// Remet un buffer dans le pool pour réutilisation
    ///
    /// Le buffer est vidé avant d'être remis dans le pool.
    /// Si le pool est plein, le buffer est simplement abandonné.
    pub fn put(&mut self, mut buf: Vec<u8>) {
        buf.clear();
        if self.buffers.len() < self.max_size {
            self.buffers.push_back(buf);
        }
    }

    /// Retourne le nombre de buffers disponibles
    pub fn available(&self) -> usize {
        self.buffers.len()
    }

    /// Retourne la capacité par défaut des buffers
    pub fn default_capacity(&self) -> usize {
        self.default_capacity
    }

    /// Retourne la taille maximale du pool
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

impl Default for BufferPool {
    /// Crée un pool avec des paramètres par défaut :
    /// - 16 buffers maximum
    /// - 64 Ko par buffer
    fn default() -> Self {
        Self::new(16, 64 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = BufferPool::new(8, 1024);
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.default_capacity(), 1024);
        assert_eq!(pool.max_size(), 8);
    }

    #[test]
    fn test_pool_default() {
        let pool = BufferPool::default();
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.default_capacity(), 64 * 1024);
        assert_eq!(pool.max_size(), 16);
    }

    #[test]
    fn test_get_and_put() {
        let mut pool = BufferPool::new(4, 512);

        // Récupérer un buffer (nouveau)
        let mut buf1 = pool.get();
        assert_eq!(buf1.capacity(), 512);
        assert!(buf1.is_empty());

        // Remplir et remettre
        buf1.extend_from_slice(b"test");
        pool.put(buf1);
        assert_eq!(pool.available(), 1);

        // Récupérer un buffer (réutilisé)
        let buf2 = pool.get();
        assert_eq!(pool.available(), 0);
        // Le buffer doit être vidé
        assert!(buf2.is_empty());
    }

    #[test]
    fn test_pool_capacity_limit() {
        let mut pool = BufferPool::new(2, 256);

        // Remplir le pool
        let buf1 = pool.get();
        let buf2 = pool.get();
        pool.put(buf1);
        pool.put(buf2);
        assert_eq!(pool.available(), 2);

        // Ajouter un buffer de trop
        let buf3 = pool.get();
        pool.put(buf3);
        // Le pool ne doit pas dépasser 2
        assert_eq!(pool.available(), 2);
    }
}
