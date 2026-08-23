//! Module d'allocation mémoire GPU
//!
//! Ce module fournit des allocations optimisées pour la mémoire GPU
//! avec suivi des statistiques et gestion des pools.

mod gpu_allocator;
mod pool;
mod stats;

pub use gpu_allocator::{GpuAllocator, GpuAllocatorBlock};
pub use pool::GpuPoolAllocator;
pub use stats::{AllocationStats, PoolStats};

#[cfg(test)]
mod tests;
