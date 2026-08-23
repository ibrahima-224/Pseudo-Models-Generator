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

//! Configuration de la structure et force structurelle.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md` §5.6.
//! Ce module définit le paramètre `structure_strength` (0.0 à 1.0) qui contrôle
//! l'intensité des structures appliquées aux tenseurs.
//!
//! ## Propriétés
//!
//! - `structure_strength = 0.0` : pas de structure (modèle de base) ;
//! - `structure_strength = 1.0` : structure maximale ;
//! - Valeurs intermédiaires : interpolation linéaire.

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// Paramètre de force structurelle (0.0 à 1.0).
///
/// Ce paramètre contrôle l'intensité des structures appliquées aux tenseurs.
/// Il est utilisé par l'injecteur pour mélanger le modèle de base et la
/// structure cible.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructureStrength(f64);

impl StructureStrength {
    /// Crée un nouveau paramètre de force structurelle.
    ///
    /// # Entrées
    /// - `value` : valeur entre 0.0 et 1.0.
    ///
    /// # Erreurs
    /// [`CoreError::Validation`] si la valeur est hors [0.0, 1.0].
    pub fn new(value: f64) -> CoreResult<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CoreError::Validation(format!(
                "force structurelle hors [0.0, 1.0] : {value}"
            )));
        }
        Ok(Self(value))
    }

    /// Retourne la valeur numérique.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Vérifie si la structure est désactivée (force = 0.0).
    pub fn is_disabled(&self) -> bool {
        self.0 == 0.0
    }

    /// Vérifie si la structure est maximale (force = 1.0).
    pub fn is_maximal(&self) -> bool {
        self.0 == 1.0
    }
}

/// Configuration de structure pour un tenseur.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureConfig {
    /// Force structurelle (0.0 à 1.0).
    strength: StructureStrength,
    /// Type de structure (optionnel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    structure_type: Option<String>,
}

impl StructureConfig {
    /// Crée une nouvelle configuration de structure.
    ///
    /// # Entrées
    /// - `strength` : force structurelle.
    pub fn new(strength: StructureStrength) -> Self {
        Self {
            strength,
            structure_type: None,
        }
    }

    /// Crée une configuration avec un type de structure spécifique.
    ///
    /// # Entrées
    /// - `strength` : force structurelle ;
    /// - `structure_type` : nom du type de structure.
    pub fn with_type(strength: StructureStrength, structure_type: &str) -> Self {
        Self {
            strength,
            structure_type: Some(structure_type.to_string()),
        }
    }

    /// Retourne la force structurelle.
    pub fn strength(&self) -> StructureStrength {
        self.strength
    }

    /// Retourne le type de structure s'il est défini.
    pub fn structure_type(&self) -> Option<&str> {
        self.structure_type.as_deref()
    }

    /// Vérifie si la structure est désactivée.
    pub fn is_disabled(&self) -> bool {
        self.strength.is_disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_strength_new_valid() {
        let s = StructureStrength::new(0.5);
        assert!(s.is_ok());
        assert_eq!(s.unwrap().value(), 0.5);
    }

    #[test]
    fn structure_strength_new_invalid() {
        let s = StructureStrength::new(1.5);
        assert!(s.is_err());
    }

    #[test]
    fn structure_strength_boundaries() {
        let s0 = StructureStrength::new(0.0).unwrap();
        assert!(s0.is_disabled());
        assert!(!s0.is_maximal());

        let s1 = StructureStrength::new(1.0).unwrap();
        assert!(!s1.is_disabled());
        assert!(s1.is_maximal());
    }

    #[test]
    fn structure_config_new() {
        let strength = StructureStrength::new(0.7).unwrap();
        let config = StructureConfig::new(strength);
        assert_eq!(config.strength().value(), 0.7);
        assert!(config.structure_type().is_none());
    }

    #[test]
    fn structure_config_with_type() {
        let strength = StructureStrength::new(0.3).unwrap();
        let config = StructureConfig::with_type(strength, "low_rank");
        assert_eq!(config.structure_type(), Some("low_rank"));
    }

    #[test]
    fn serde_roundtrip() {
        let strength = StructureStrength::new(0.7).unwrap();
        let config = StructureConfig::with_type(strength, "correlation");
        let json = serde_json::to_string(&config).unwrap();
        let restored: StructureConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.strength().value(), restored.strength().value());
        assert_eq!(config.structure_type(), restored.structure_type());
    }
}
