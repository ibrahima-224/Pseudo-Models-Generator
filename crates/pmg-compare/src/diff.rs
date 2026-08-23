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

//! Format unifié de différences — représentation standardisée des écarts.
//!
//! Ce module définit les types pour représenter les différences entre
//! deux modèles de manière uniforme, avec les indicateurs :
//! - `+` pour les ajouts
//! - `-` pour les suppressions
//! - `~` pour les modifications
//!
//! # Responsabilités
//!
//! - Structure `Diff` pour représenter une différence unique ;
//! - Énumération `DiffType` pour classifier les types de différences ;
//! - Méthodes d'affichage formatées avec les icônes appropriées.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les structures sont conçues pour être immuables après construction.

/// Type de différence entre deux éléments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffType {
    /// Élément ajouté dans le modèle comparé (+).
    Added,
    /// Élément supprimé du modèle original (-).
    Removed,
    /// Élément modifié entre les deux modèles (~).
    Modified,
}

impl std::fmt::Display for DiffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffType::Added => write!(f, "+"),
            DiffType::Removed => write!(f, "-"),
            DiffType::Modified => write!(f, "~"),
        }
    }
}

/// Représente une différence unique entre deux modèles.
///
/// # Exemple
///
/// ```
/// use pmg_compare::diff::{Diff, DiffType};
///
/// let diff = Diff::new(
///     DiffType::Modified,
///     "model.layers.0.mlp.gate.weight".to_string(),
///     Some("[4096, 4096]".to_string()),
///     Some("[4096, 2048]".to_string()),
///     "Shape modifiée".to_string(),
/// );
///
/// assert_eq!(diff.diff_type, DiffType::Modified);
/// assert_eq!(diff.path, "model.layers.0.mlp.gate.weight");
/// ```
#[derive(Debug, Clone)]
pub struct Diff {
    /// Type de la différence.
    pub diff_type: DiffType,
    /// Chemin ou nom de l'élément concerné.
    pub path: String,
    /// Valeur dans le modèle original (si applicable).
    pub original_value: Option<String>,
    /// Valeur dans le modèle comparé (si applicable).
    pub compared_value: Option<String>,
    /// Description détaillée de la différence.
    pub description: String,
}

impl Diff {
    /// Crée une nouvelle différence.
    pub fn new(
        diff_type: DiffType,
        path: String,
        original_value: Option<String>,
        compared_value: Option<String>,
        description: String,
    ) -> Self {
        Self {
            diff_type,
            path,
            original_value,
            compared_value,
            description,
        }
    }

    /// Crée une différence d'ajout.
    pub fn added(path: String, value: String, description: String) -> Self {
        Self::new(DiffType::Added, path, None, Some(value), description)
    }

    /// Crée une différence de suppression.
    pub fn removed(path: String, value: String, description: String) -> Self {
        Self::new(DiffType::Removed, path, Some(value), None, description)
    }

    /// Crée une différence de modification.
    pub fn modified(path: String, original: String, compared: String, description: String) -> Self {
        Self::new(
            DiffType::Modified,
            path,
            Some(original),
            Some(compared),
            description,
        )
    }
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.diff_type {
            DiffType::Added => {
                write!(
                    f,
                    "+ {} : {} = {}",
                    self.path,
                    self.description,
                    self.compared_value.as_deref().unwrap_or("N/A")
                )
            },
            DiffType::Removed => {
                write!(
                    f,
                    "- {} : {} = {}",
                    self.path,
                    self.description,
                    self.original_value.as_deref().unwrap_or("N/A")
                )
            },
            DiffType::Modified => {
                write!(
                    f,
                    "~ {} : {} ({} → {})",
                    self.path,
                    self.description,
                    self.original_value.as_deref().unwrap_or("N/A"),
                    self.compared_value.as_deref().unwrap_or("N/A")
                )
            },
        }
    }
}

/// Collection de différences avec des méthodes utilitaires.
#[derive(Debug, Clone, Default)]
pub struct DiffCollection {
    /// Liste des différences.
    pub diffs: Vec<Diff>,
}

impl DiffCollection {
    /// Crée une nouvelle collection vide.
    pub fn new() -> Self {
        Self { diffs: Vec::new() }
    }

    /// Ajoute une différence à la collection.
    pub fn add(&mut self, diff: Diff) {
        self.diffs.push(diff);
    }

    /// Retourne le nombre de différences.
    pub fn len(&self) -> usize {
        self.diffs.len()
    }

    /// Vérifie si la collection est vide.
    pub fn is_empty(&self) -> bool {
        self.diffs.is_empty()
    }

    /// Retourne uniquement les ajouts.
    pub fn additions(&self) -> Vec<&Diff> {
        self.diffs
            .iter()
            .filter(|d| d.diff_type == DiffType::Added)
            .collect()
    }

    /// Retourne uniquement les suppressions.
    pub fn removals(&self) -> Vec<&Diff> {
        self.diffs
            .iter()
            .filter(|d| d.diff_type == DiffType::Removed)
            .collect()
    }

    /// Retourne uniquement les modifications.
    pub fn modifications(&self) -> Vec<&Diff> {
        self.diffs
            .iter()
            .filter(|d| d.diff_type == DiffType::Modified)
            .collect()
    }

    /// Vérifie s'il y a des ajouts.
    pub fn has_additions(&self) -> bool {
        self.diffs.iter().any(|d| d.diff_type == DiffType::Added)
    }

    /// Vérifie s'il y a des suppressions.
    pub fn has_removals(&self) -> bool {
        self.diffs.iter().any(|d| d.diff_type == DiffType::Removed)
    }

    /// Vérifie s'il y a des modifications.
    pub fn has_modifications(&self) -> bool {
        self.diffs.iter().any(|d| d.diff_type == DiffType::Modified)
    }
}

impl std::fmt::Display for DiffCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            writeln!(f, "Aucune différence détectée.")?;
        } else {
            writeln!(f, "Différences détectées ({}):", self.len())?;
            for diff in &self.diffs {
                writeln!(f, "  {}", diff)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_type_display() {
        assert_eq!(DiffType::Added.to_string(), "+");
        assert_eq!(DiffType::Removed.to_string(), "-");
        assert_eq!(DiffType::Modified.to_string(), "~");
    }

    #[test]
    fn diff_creation() {
        let diff = Diff::added(
            "vocab_size".to_string(),
            "32000".to_string(),
            "Paramètre ajouté".to_string(),
        );

        assert_eq!(diff.diff_type, DiffType::Added);
        assert_eq!(diff.path, "vocab_size");
        assert!(diff.original_value.is_none());
        assert_eq!(diff.compared_value, Some("32000".to_string()));
    }

    #[test]
    fn diff_collection_operations() {
        let mut collection = DiffCollection::new();

        collection.add(Diff::added(
            "param1".to_string(),
            "value1".to_string(),
            "Ajout".to_string(),
        ));

        collection.add(Diff::removed(
            "param2".to_string(),
            "value2".to_string(),
            "Suppression".to_string(),
        ));

        collection.add(Diff::modified(
            "param3".to_string(),
            "old".to_string(),
            "new".to_string(),
            "Modification".to_string(),
        ));

        assert_eq!(collection.len(), 3);
        assert!(!collection.is_empty());
        assert!(collection.has_additions());
        assert!(collection.has_removals());
        assert!(collection.has_modifications());

        assert_eq!(collection.additions().len(), 1);
        assert_eq!(collection.removals().len(), 1);
        assert_eq!(collection.modifications().len(), 1);
    }
}
