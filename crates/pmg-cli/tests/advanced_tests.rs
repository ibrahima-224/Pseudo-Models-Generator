//! Tests avancés pour la CLI PMG.
//!
//! Ce fichier organise tous les modules de tests avancés :
//! - Tests de sécurité (priorité 1)
//! - Tests d'intégration (priorité 2)
//! - Tests unitaires avancés (priorité 3)
//! - Tests de performance (priorité 4)
//! - Tests de charge (simulation de grands modèles)

// Module commun pour les utilitaires de test
mod common;

// Tests de sécurité (priorité 1)
mod security;

// Tests d'intégration (priorité 2)
mod integration;

// Tests unitaires avancés (priorité 3)
mod unit;

// Tests de performance (priorité 4)
mod performance;

// Tests de charge (simulation de grands modèles)
mod load;
