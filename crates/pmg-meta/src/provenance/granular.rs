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

//! Sous-module contenant les structures de provenance granulaire.

use std::collections::BTreeMap;

use pmg_core::origin::{Confidence, Origin};
use serde::{Deserialize, Serialize};

/// Provenance granulaire d'un tenseur spécifique.
///
/// Combine l'origine (OBSERVÉ/ESTIMÉ/GÉNÉRÉ/INCONNU) et le niveau de confiance
/// pour un tenseur donné, permettant une traçabilité fine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorProvenance {
    /// Origine de la valeur du tenseur (comment elle a été obtenue).
    pub origin: Origin,
    /// Niveau de confiance associé à la valeur.
    pub confidence: Confidence,
}

/// Provenance granulaire d'un champ de configuration spécifique.
///
/// Similaire à [`TensorProvenance`] mais pour les champs de configuration
/// (modèle, seed, distribution, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldProvenance {
    /// Origine de la valeur du champ.
    pub origin: Origin,
    /// Niveau de confiance associé à la valeur.
    pub confidence: Confidence,
}

/// Provenance granulaire complète pour un modèle.
///
/// Regroupe la provenance de tous les tenseurs et champs de configuration,
/// permettant une traçabilité fine de chaque élément du modèle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GranularProvenance {
    /// Provenance par tenseur (nom → provenance).
    pub tensor_provenance: BTreeMap<String, TensorProvenance>,
    /// Provenance par champ de configuration (nom → provenance).
    pub field_provenance: BTreeMap<String, FieldProvenance>,
}

impl GranularProvenance {
    /// Crée une provenance granulaire vide.
    pub fn new() -> Self {
        Self {
            tensor_provenance: BTreeMap::new(),
            field_provenance: BTreeMap::new(),
        }
    }

    /// Valide la cohérence de la provenance granulaire.
    ///
    /// Vérifie que :
    /// - chaque tenseur a une origine et une confiance valides ;
    /// - chaque champ a une origine et une confiance valides ;
    /// - il n'y a pas de doublons incohérents.
    pub fn validate(&self) -> Result<(), String> {
        // Vérification basique : les maps ne sont pas vides (optionnel)
        // La validation principale est la cohérence des types
        for (name, tp) in &self.tensor_provenance {
            if tp.origin == Origin::Unknown && tp.confidence == Confidence::Exact {
                return Err(format!(
                    "Incohérence pour le tenseur '{}': origine INCONNU avec confiance EXACT",
                    name
                ));
            }
        }
        for (name, fp) in &self.field_provenance {
            if fp.origin == Origin::Unknown && fp.confidence == Confidence::Exact {
                return Err(format!(
                    "Incohérence pour le champ '{}': origine INCONNU avec confiance EXACT",
                    name
                ));
            }
        }
        Ok(())
    }

    /// Retourne le nombre total d'éléments tracés.
    pub fn total_tracked(&self) -> usize {
        self.tensor_provenance.len() + self.field_provenance.len()
    }

    /// Fusionne une autre provenance granulaire dans celle-ci.
    ///
    /// Les entrées existantes sont écrasées par les nouvelles.
    pub fn merge(&mut self, other: GranularProvenance) {
        self.tensor_provenance.extend(other.tensor_provenance);
        self.field_provenance.extend(other.field_provenance);
    }
}

impl Default for GranularProvenance {
    fn default() -> Self {
        Self::new()
    }
}
