//! Kernels CUDA/PTX pour la génération de modèles
//!
//! Ce module contient les kernels PTX optimisés pour les opérations
//! intensives de génération de modèles sur GPU.
//!
//! # Organisation
//!
//! Le module est organisé en plusieurs sous-modules pour respecter la limite
//! de 500 lignes par fichier :
//!
//! - `kernel_config` : Configuration d'exécution des kernels
//! - `kernel_args` : Interfaces d'argumentation pour les kernels
//! - `kernel_core` : Noyau de compilation et exécution des kernels
//! - `kernel_ptx` : Templates de code PTX précompilés
//! - `kernel_registry` : Registre et lifecycle des kernels précompilés
//!
//! Toutes les API publiques sont réexportées ici pour maintenir la
//! rétrocompatibilité avec le code existant.

// Déclaration des sous-modules
pub mod kernel_args;
pub mod kernel_config;
pub mod kernel_core;
pub mod kernel_ptx;
pub mod kernel_registry;

// Réexports pour rétrocompatibilité API
pub use kernel_args::ToGpuArg;
pub use kernel_config::KernelConfig;
pub use kernel_core::GpuKernel;
pub use kernel_ptx::{
    BF16_CONVERSION_KERNEL, MATRIX_MULTIPLICATION_KERNEL, MIXTURE_DISTRIBUTION_KERNEL,
    NORMAL_GENERATION_KERNEL,
};
pub use kernel_registry::KernelRegistry;

// Les tests ont été déplacés vers les modules respectifs
// pour maintenir la modularité et la lisibilité
