//! Module de tests de sécurité pour la CLI PMG.
//!
//! Ce module contient les tests de sécurité critiques :
//! - Tests d'injection d'arguments malveillants
//! - Tests de parcours de fichiers (path traversal)
//! - Tests de validation des entrées invalides

pub mod injection_tests;
pub mod input_validation_tests;
pub mod path_traversal_tests;
