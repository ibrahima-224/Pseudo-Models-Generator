//! Module commun pour les tests de la CLI PMG.
//!
//! Ce module fournit des utilitaires partagés pour tous les tests :
//! - Fonctions d'aide pour l'exécution de commandes
//! - Fixtures et données de test prédéfinies
//! - Gestion des répertoires temporaires

pub mod fixtures;
pub mod helpers;

// Réexportation des éléments les plus utilisés pour simplifier les imports
pub use fixtures::*;
pub use helpers::*;
