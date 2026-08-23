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

//! Provenance et confiance des valeurs de métadonnées (type transverse).
//!
//! Règle architecturale : toute valeur importante d'un modèle porte une
//! `origin` et une `confidence`. Ces catégories alimentent `espec`, le
//! manifeste et la politique d'honnêteté scientifique (Zero-Payload).
//! Référence : `docs/architecture/03-modeles-de-donnees.md` §1.

use serde::{Deserialize, Serialize};

/// Provenance d'une valeur : comment PMG l'a obtenue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Origin {
    /// OBSERVÉ : lu dans un artefact autorisé (config/index/header).
    Observed,
    /// ESTIMÉ : calculé mathématiquement depuis des données observées.
    Derived,
    /// GÉNÉRÉ : produit par le générateur PMG.
    Generated,
    /// INCONNU : non établissable.
    Unknown,
}

impl Origin {
    /// Libellé français stable pour l'affichage (`espec`, rapports).
    pub fn label_fr(self) -> &'static str {
        match self {
            Origin::Observed => "OBSERVÉ",
            Origin::Derived => "ESTIMÉ",
            Origin::Generated => "GÉNÉRÉ",
            Origin::Unknown => "INCONNU",
        }
    }
}

/// Niveau de confiance associé à une valeur.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Confidence {
    /// EXACT : information certaine (lue directement).
    Exact,
    /// DERIVED : déduction exacte à partir d'observations.
    Derived,
    /// ESTIMATED : estimation avec incertitude.
    Estimated,
    /// SYNTHETIC : valeur générée (jamais présentée comme mesurée).
    Synthetic,
    /// UNKNOWN : inconnue.
    Unknown,
}

impl Confidence {
    /// Libellé anglais stable (normalisé dans les artefacts de sortie).
    pub fn label(self) -> &'static str {
        match self {
            Confidence::Exact => "EXACT",
            Confidence::Derived => "DERIVED",
            Confidence::Estimated => "ESTIMATED",
            Confidence::Synthetic => "SYNTHETIC",
            Confidence::Unknown => "UNKNOWN",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Confidence, Origin};

    #[test]
    fn origin_labels_are_french() {
        // Les libellés français sont la référence d'affichage.
        assert_eq!(Origin::Observed.label_fr(), "OBSERVÉ");
        assert_eq!(Origin::Derived.label_fr(), "ESTIMÉ");
        assert_eq!(Origin::Generated.label_fr(), "GÉNÉRÉ");
        assert_eq!(Origin::Unknown.label_fr(), "INCONNU");
    }

    #[test]
    fn confidence_labels_are_normalized() {
        // Les libellés anglais sont la forme sérialisée stable.
        assert_eq!(Confidence::Exact.label(), "EXACT");
        assert_eq!(Confidence::Unknown.label(), "UNKNOWN");
    }

    #[test]
    fn serde_roundtrip() {
        // Les catégories doivent être sérialisables (manifeste, provenance.json).
        let orig = Origin::Observed;
        let json = serde_json::to_string(&orig).unwrap();
        assert_eq!(json, "\"Observed\"");
        assert_eq!(serde_json::from_str::<Origin>(&json).unwrap(), orig);
    }
}
