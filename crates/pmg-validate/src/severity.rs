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

//! Niveaux de sévérité pour les résultats de validation.
//!
//! Ce module définit les différents niveaux de sévérité utilisés
//! dans les rapports de validation des pseudo-modèles.
//!
//! # Niveaux
//!
//! - `INFO` : information pure, pas d'action requise ;
//! - `WARNING` : avertissement, action recommandée mais non bloquante ;
//! - `ERROR` : erreur, action requise mais le processus peut continuer ;
//! - `CRITICAL` : erreur critique, le processus doit s'arrêter.

use std::fmt;

/// Niveau de sévérité d'un message de validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Information pure, pas d'action requise.
    Info,
    /// Avertissement, action recommandée mais non bloquante.
    Warning,
    /// Erreur, action requise mais le processus peut continuer.
    Error,
    /// Erreur critique, le processus doit s'arrêter.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl Severity {
    /// Retourne `true` si le niveau est `Warning` ou plus grave.
    pub fn is_warning_or_worse(&self) -> bool {
        *self >= Severity::Warning
    }

    /// Retourne `true` si le niveau est `Error` ou plus grave.
    pub fn is_error_or_worse(&self) -> bool {
        *self >= Severity::Error
    }

    /// Retourne `true` si le niveau est `Critical`.
    pub fn is_critical(&self) -> bool {
        *self == Severity::Critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn severity_display() {
        assert_eq!(Severity::Info.to_string(), "INFO");
        assert_eq!(Severity::Warning.to_string(), "WARNING");
        assert_eq!(Severity::Error.to_string(), "ERROR");
        assert_eq!(Severity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn severity_methods() {
        assert!(!Severity::Info.is_warning_or_worse());
        assert!(Severity::Warning.is_warning_or_worse());
        assert!(Severity::Error.is_warning_or_worse());
        assert!(Severity::Critical.is_warning_or_worse());

        assert!(!Severity::Info.is_error_or_worse());
        assert!(!Severity::Warning.is_error_or_worse());
        assert!(Severity::Error.is_error_or_worse());
        assert!(Severity::Critical.is_error_or_worse());

        assert!(!Severity::Info.is_critical());
        assert!(!Severity::Warning.is_critical());
        assert!(!Severity::Error.is_critical());
        assert!(Severity::Critical.is_critical());
    }
}
