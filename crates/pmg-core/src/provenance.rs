// Copyright (C) 2024 PMG Contributors
// This file is part of PMG (Pseudo-Model Generator).
//
// PMG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PMG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PMG.  If not, see <https://www.gnu.org/licenses/>.

//! Module de provenance pour tracer la source des tenseurs générés.
//!
//! Ce module définit le type `Provenance` qui combine une origine (`ProvenanceOrigin`)
//! et un niveau de confiance (`f64` entre 0.0 et 1.0). La provenance est utilisée
//! pour maintenir la traçabilité des valeurs dans le système PMG.
//!
//! Référence : `docs/architecture/03-modeles-de-donnees.md` §3.6

use serde::{Deserialize, Serialize};
use std::fmt;

/// Erreurs spécifiques à la validation de la provenance.
#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceError {
    /// La confiance doit être entre 0.0 et 1.0 (inclus).
    InvalidConfidence(f64),
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenanceError::InvalidConfidence(value) => {
                write!(
                    f,
                    "La confiance doit être entre 0.0 et 1.0, valeur reçue : {}",
                    value
                )
            },
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// Origine de la provenance d'une valeur.
///
/// Cette enumération catégorise la source d'une valeur selon son type
/// de génération ou d'observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ProvenanceOrigin {
    /// Calculé mathématiquement à partir de données observées.
    Math,
    /// Produit par un modèle de génération.
    Model,
    /// Combinaison de sources mathématiques et modélisées.
    Hybrid,
    /// Origine inconnue ou non déterminée.
    #[default]
    Unknown,
}

impl ProvenanceOrigin {
    /// Libellé français stable pour l'affichage.
    pub fn label_fr(self) -> &'static str {
        match self {
            ProvenanceOrigin::Math => "MATHÉMATIQUE",
            ProvenanceOrigin::Model => "MODÉLISÉ",
            ProvenanceOrigin::Hybrid => "HYBRIDE",
            ProvenanceOrigin::Unknown => "INCONNU",
        }
    }

    /// Libellé anglais normalisé pour la sérialisation.
    pub fn label_en(self) -> &'static str {
        match self {
            ProvenanceOrigin::Math => "MATH",
            ProvenanceOrigin::Model => "MODEL",
            ProvenanceOrigin::Hybrid => "HYBRID",
            ProvenanceOrigin::Unknown => "UNKNOWN",
        }
    }
}

impl fmt::Display for ProvenanceOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label_en())
    }
}

/// Structure représentant la provenance complète d'une valeur.
///
/// Combine une origine (`ProvenanceOrigin`) et un niveau de confiance
/// (`confidence: f64`) entre 0.0 et 1.0. La confiance 1.0 indique une
/// certitude absolue, tandis que 0.0 indique une incertitude totale.
///
/// # Exemple
///
/// ```rust
/// use pmg_core::provenance::{Provenance, ProvenanceOrigin};
///
/// // Création d'une provenance avec confiance maximale
/// let prov = Provenance::new(ProvenanceOrigin::Math, 1.0).unwrap();
/// assert_eq!(prov.origin(), ProvenanceOrigin::Math);
/// assert_eq!(prov.confidence(), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Origine de la valeur.
    origin: ProvenanceOrigin,
    /// Niveau de confiance entre 0.0 et 1.0.
    confidence: f64,
}

impl Provenance {
    /// Crée une nouvelle instance de `Provenance` avec validation.
    ///
    /// # Arguments
    ///
    /// * `origin` - L'origine de la valeur.
    /// * `confidence` - Le niveau de confiance (doit être entre 0.0 et 1.0).
    ///
    /// # Retour
    ///
    /// Retourne `Ok(Provenance)` si la confiance est valide, sinon `Err(ProvenanceError)`.
    pub fn new(origin: ProvenanceOrigin, confidence: f64) -> Result<Self, ProvenanceError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(ProvenanceError::InvalidConfidence(confidence));
        }
        Ok(Provenance { origin, confidence })
    }

    /// Retourne l'origine de la provenance.
    pub fn origin(&self) -> ProvenanceOrigin {
        self.origin
    }

    /// Retourne le niveau de confiance.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Met à jour le niveau de confiance avec validation.
    ///
    /// # Arguments
    ///
    /// * `new_confidence` - Le nouveau niveau de confiance.
    ///
    /// # Retour
    ///
    /// Retourne `Ok(())` si la confiance est valide, sinon `Err(ProvenanceError)`.
    pub fn set_confidence(&mut self, new_confidence: f64) -> Result<(), ProvenanceError> {
        if !(0.0..=1.0).contains(&new_confidence) {
            return Err(ProvenanceError::InvalidConfidence(new_confidence));
        }
        self.confidence = new_confidence;
        Ok(())
    }

    /// Vérifie si la provenance est cohérente (confiance > 0.0 pour origine non-Unknown).
    pub fn is_consistent(&self) -> bool {
        match self.origin {
            ProvenanceOrigin::Unknown => true, // Unknown peut avoir n'importe quelle confiance
            _ => self.confidence > 0.0,
        }
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Provenance {
            origin: ProvenanceOrigin::Unknown,
            confidence: 0.0,
        }
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Provenance {{ origine: {}, confiance: {:.2} }}",
            self.origin.label_fr(),
            self.confidence
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_creation_valide() {
        // Test de création avec des valeurs valides
        let prov = Provenance::new(ProvenanceOrigin::Math, 0.75);
        assert!(prov.is_ok());
        let prov = prov.unwrap();
        assert_eq!(prov.origin(), ProvenanceOrigin::Math);
        assert_eq!(prov.confidence(), 0.75);
    }

    #[test]
    fn test_provenance_confidence_limite_inferieure() {
        // Test avec confiance minimale (0.0)
        let prov = Provenance::new(ProvenanceOrigin::Model, 0.0);
        assert!(prov.is_ok());
        assert_eq!(prov.unwrap().confidence(), 0.0);
    }

    #[test]
    fn test_provenance_confidence_limite_superieure() {
        // Test avec confiance maximale (1.0)
        let prov = Provenance::new(ProvenanceOrigin::Hybrid, 1.0);
        assert!(prov.is_ok());
        assert_eq!(prov.unwrap().confidence(), 1.0);
    }

    #[test]
    fn test_provenance_confidence_hors_limites() {
        // Test avec confiance invalide (négative)
        let prov = Provenance::new(ProvenanceOrigin::Unknown, -0.1);
        assert!(prov.is_err());
        assert_eq!(prov.unwrap_err(), ProvenanceError::InvalidConfidence(-0.1));

        // Test avec confiance invalide (> 1.0)
        let prov = Provenance::new(ProvenanceOrigin::Math, 1.5);
        assert!(prov.is_err());
        assert_eq!(prov.unwrap_err(), ProvenanceError::InvalidConfidence(1.5));
    }

    #[test]
    fn test_provenance_default() {
        // Test des valeurs par défaut
        let prov = Provenance::default();
        assert_eq!(prov.origin(), ProvenanceOrigin::Unknown);
        assert_eq!(prov.confidence(), 0.0);
    }

    #[test]
    fn test_provenance_origin_labels() {
        // Test des libellés français
        assert_eq!(ProvenanceOrigin::Math.label_fr(), "MATHÉMATIQUE");
        assert_eq!(ProvenanceOrigin::Model.label_fr(), "MODÉLISÉ");
        assert_eq!(ProvenanceOrigin::Hybrid.label_fr(), "HYBRIDE");
        assert_eq!(ProvenanceOrigin::Unknown.label_fr(), "INCONNU");

        // Test des libellés anglais
        assert_eq!(ProvenanceOrigin::Math.label_en(), "MATH");
        assert_eq!(ProvenanceOrigin::Model.label_en(), "MODEL");
        assert_eq!(ProvenanceOrigin::Hybrid.label_en(), "HYBRID");
        assert_eq!(ProvenanceOrigin::Unknown.label_en(), "UNKNOWN");
    }

    #[test]
    fn test_provenance_set_confidence() {
        // Test de mise à jour de la confiance
        let mut prov = Provenance::new(ProvenanceOrigin::Math, 0.5).unwrap();
        assert!(prov.set_confidence(0.8).is_ok());
        assert_eq!(prov.confidence(), 0.8);

        // Test de mise à jour avec valeur invalide
        assert!(prov.set_confidence(2.0).is_err());
        assert_eq!(prov.confidence(), 0.8); // Inchangé
    }

    #[test]
    fn test_provenance_coherence() {
        // Test de cohérence
        let prov_unknown = Provenance::new(ProvenanceOrigin::Unknown, 0.0).unwrap();
        assert!(prov_unknown.is_consistent()); // Unknown est toujours cohérent

        let prov_math_zero = Provenance::new(ProvenanceOrigin::Math, 0.0).unwrap();
        assert!(!prov_math_zero.is_consistent()); // Math avec confiance 0 n'est pas cohérent

        let prov_math_nonzero = Provenance::new(ProvenanceOrigin::Math, 0.1).unwrap();
        assert!(prov_math_nonzero.is_consistent()); // Math avec confiance > 0 est cohérent
    }

    #[test]
    fn test_provenance_serialization() {
        // Test de sérialisation/désérialisation
        let prov = Provenance::new(ProvenanceOrigin::Hybrid, 0.42).unwrap();
        let json = serde_json::to_string(&prov).unwrap();
        let deserialized: Provenance = serde_json::from_str(&json).unwrap();
        assert_eq!(prov, deserialized);
    }

    #[test]
    fn test_provenance_display() {
        // Test de l'affichage
        let prov = Provenance::new(ProvenanceOrigin::Model, 0.75).unwrap();
        let display = format!("{}", prov);
        assert!(display.contains("MODÉLISÉ"));
        assert!(display.contains("0.75"));
    }
}
