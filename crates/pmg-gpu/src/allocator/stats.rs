//! Statistiques d'allocation GPU
//!
//! Ce module définit les structures de données pour le suivi des statistiques
//! d'allocation mémoire GPU.

/// Statistiques d'allocation
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Nombre total d'allocations effectuées
    pub total_allocations: usize,
    /// Nombre total de libérations effectuées
    pub total_deallocations: usize,
    /// Nombre total d'octets alloués
    pub total_bytes_allocated: u64,
    /// Nombre total d'octets libérés
    pub total_bytes_deallocated: u64,
    /// Nombre d'allocations actives
    pub active_allocations: usize,
    /// Nombre d'octets actuellement alloués
    pub active_bytes: usize,
    /// Taille maximale d'une allocation
    pub max_allocation_size: usize,
    /// Taille minimale d'une allocation
    pub min_allocation_size: usize,
}

/// Statistiques du pool d'allocation
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Nombre total de blocs dans le pool
    pub total_blocks: usize,
    /// Mémoire totale dans le pool (en octets)
    pub total_memory: usize,
    /// Taille minimale des blocs
    pub min_block_size: usize,
    /// Taille maximale des blocs
    pub max_block_size: usize,
}
