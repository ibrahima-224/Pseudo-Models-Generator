//! Templates de code PTX précompilés pour les kernels GPU
//!
//! Ce module contient les constantes de code PTX pour les kernels de génération
//! de modèles. Ces kernels sont optimisés pour les opérations intensives
//! comme la génération de nombres normaux, la conversion BF16 et les mélanges.
//!
//! # Organisation
//!
//! Les kernels sont répartis en deux sous-modules pour respecter la limite
//! de 300 lignes par fichier (bas-niveau) :
//!
//! - `kernel_ptx_generation` : Kernels de génération et conversion
//! - `kernel_ptx_advanced` : Kernels avancés (mélange, matriciel)

// Déclaration des sous-modules
pub mod kernel_ptx_advanced;
pub mod kernel_ptx_generation;

// Réexports pour rétrocompatibilité API
pub use kernel_ptx_advanced::{MATRIX_MULTIPLICATION_KERNEL, MIXTURE_DISTRIBUTION_KERNEL};
pub use kernel_ptx_generation::{BF16_CONVERSION_KERNEL, NORMAL_GENERATION_KERNEL};
