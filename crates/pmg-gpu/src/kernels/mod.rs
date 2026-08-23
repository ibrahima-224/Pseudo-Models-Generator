//! Implémentations GPU pour les opérations accélérables
//!
//! Ce module contient les implémentations concrètes du trait `GpuAccelerated`
//! pour diverses opérations de génération de modèles. Chaque implémentation
//! fournit à la fois un fallback CPU et une accélération GPU lorsque possible.
//!
//! # Organisation
//!
//! - `normal_generation` : Génération de distributions normales
//! - `optimized` : Kernels optimisés pour les shapes courantes en ML
//!
//! # Feature Flags
//!
//! Les implémentations GPU sont conditionnées par la feature `cuda`.
//! Sans cette feature, seules les implémentations CPU sont disponibles.

pub mod normal_generation;
pub mod optimized;

pub use normal_generation::NormalGenerationAccelerated;
pub use optimized::{BatchGenerator, CommonShapes, OptimizedGenerator};
