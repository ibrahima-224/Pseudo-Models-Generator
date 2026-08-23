//! Pool de buffers réutilisables pour l'écriture SafeTensors
//!
//! Ce module fournit un pool de buffers optimisé pour réutiliser les allocations
//! mémoire lors de l'écriture par chunks, réduisant ainsi la pression sur le
//! garbage collector et améliorant les performances.

/// Taille minimale du chunk (1 Mo).
pub const MIN_CHUNK_SIZE: usize = 1024 * 1024;

/// Taille maximale du chunk (32 Mo).
pub const MAX_CHUNK_SIZE: usize = 32 * 1024 * 1024;

/// Taille maximale du pool de buffers (32 Mo par défaut).
pub const DEFAULT_MAX_POOL_MEMORY: usize = 32 * 1024 * 1024;
