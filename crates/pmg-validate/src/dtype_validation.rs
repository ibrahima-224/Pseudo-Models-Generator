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

//! Validation des types de données (dtype) des tenseurs.
//!
//! Ce module vérifie la cohérence entre le dtype déclaré dans le blueprint
//! et le dtype observé dans le modèle généré.
//!
//! # Responsabilités
//!
//! - Comparaison des dtypes observés vs attendus ;
//! - Vérification de la compatibilité des types pour les opérations ;
//! - Détection des incohérences de type.
//!
//! # Formule
//!
//! La validation est simple : `dtype_observé == dtype_attendu`.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::types::{ValidationCategory, ValidationIssue};

/// Représente un type de données simplifié pour la validation.
/// Utilisé quand on ne veut pas dépendre directement de pmg-core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimpleDType {
    /// Flottant 64 bits.
    F64,
    /// Flottant 32 bits.
    F32,
    /// Flottant 16 bits.
    F16,
    /// Bfloat16.
    Bf16,
    /// FP8 e4m3.
    F8E4M3,
    /// FP8 e5m2.
    F8E5M2,
    /// Entier signé 64 bits.
    I64,
    /// Entier signé 32 bits.
    I32,
    /// Entier signé 16 bits.
    I16,
    /// Entier signé 8 bits.
    I8,
    /// Entier non signé 64 bits.
    U64,
    /// Entier non signé 32 bits.
    U32,
    /// Entier non signé 16 bits.
    U16,
    /// Entier non signé 8 bits.
    U8,
    /// Booléen.
    Bool,
}

impl SimpleDType {
    /// Nombre d'octets par élément.
    pub fn size_bytes(self) -> u64 {
        match self {
            SimpleDType::F64 | SimpleDType::I64 | SimpleDType::U64 => 8,
            SimpleDType::F32 | SimpleDType::I32 | SimpleDType::U32 => 4,
            SimpleDType::F16 | SimpleDType::Bf16 | SimpleDType::I16 | SimpleDType::U16 => 2,
            SimpleDType::F8E4M3
            | SimpleDType::F8E5M2
            | SimpleDType::I8
            | SimpleDType::U8
            | SimpleDType::Bool => 1,
        }
    }

    /// Vérifie si le dtype est un type flottant.
    pub fn is_floating(self) -> bool {
        matches!(
            self,
            SimpleDType::F64
                | SimpleDType::F32
                | SimpleDType::F16
                | SimpleDType::Bf16
                | SimpleDType::F8E4M3
                | SimpleDType::F8E5M2
        )
    }

    /// Vérifie si le dtype est un type entier.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            SimpleDType::I64
                | SimpleDType::I32
                | SimpleDType::I16
                | SimpleDType::I8
                | SimpleDType::U64
                | SimpleDType::U32
                | SimpleDType::U16
                | SimpleDType::U8
        )
    }
}

impl std::fmt::Display for SimpleDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimpleDType::F64 => write!(f, "F64"),
            SimpleDType::F32 => write!(f, "F32"),
            SimpleDType::F16 => write!(f, "F16"),
            SimpleDType::Bf16 => write!(f, "BF16"),
            SimpleDType::F8E4M3 => write!(f, "F8_E4M3"),
            SimpleDType::F8E5M2 => write!(f, "F8_E5M2"),
            SimpleDType::I64 => write!(f, "I64"),
            SimpleDType::I32 => write!(f, "I32"),
            SimpleDType::I16 => write!(f, "I16"),
            SimpleDType::I8 => write!(f, "I8"),
            SimpleDType::U64 => write!(f, "U64"),
            SimpleDType::U32 => write!(f, "U32"),
            SimpleDType::U16 => write!(f, "U16"),
            SimpleDType::U8 => write!(f, "U8"),
            SimpleDType::Bool => write!(f, "BOOL"),
        }
    }
}

/// Résultat de la validation de dtype pour un tenseur.
#[derive(Debug, Clone)]
pub struct DTypeValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// DType observé.
    pub observed_dtype: SimpleDType,
    /// DType attendu.
    pub expected_dtype: SimpleDType,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Valide le dtype d'un tenseur.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_dtype, SimpleDType, Severity};
///
/// // Validation d'un tenseur avec le bon dtype
/// let result = validate_dtype(
///     "layer1.weight",
///     SimpleDType::F32,
///     SimpleDType::F32,
/// );
/// assert!(result.issues.is_empty());
///
/// // Validation d'un tenseur avec un dtype incorrect
/// let result2 = validate_dtype(
///     "layer1.weight",
///     SimpleDType::F16,
///     SimpleDType::F32,
/// );
/// assert!(!result2.issues.is_empty());
/// assert_eq!(result2.issues[0].severity, Severity::Error);
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `observed_dtype` : dtype observé dans le modèle ;
/// - `expected_dtype` : dtype attendu selon le blueprint.
///
/// # Sorties
/// Un [`DTypeValidationResult`] contenant les issues détectées.
pub fn validate_dtype(
    tensor_path: &str,
    observed_dtype: SimpleDType,
    expected_dtype: SimpleDType,
) -> DTypeValidationResult {
    let mut issues = Vec::new();

    // Vérification de la correspondance des dtypes
    if observed_dtype != expected_dtype {
        issues.push(ValidationIssue {
            category: ValidationCategory::Structural,
            severity: Severity::Error,
            message: format!(
                "DType observé ({}) ne correspond pas au dtype attendu ({})",
                observed_dtype, expected_dtype
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    DTypeValidationResult {
        path: tensor_path.to_string(),
        observed_dtype,
        expected_dtype,
        issues,
    }
}

/// Valide la compatibilité des dtypes pour une opération.
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `dtype_a` : premier dtype ;
/// - `dtype_b` : second dtype ;
/// - `operation` : nom de l'opération.
///
/// # Sorties
/// Un vecteur d'[`ValidationIssue`] contenant les incompatibilités.
pub fn validate_dtype_compatibility(
    tensor_path: &str,
    dtype_a: SimpleDType,
    dtype_b: SimpleDType,
    operation: &str,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Règles de base de compatibilité
    match operation {
        "add" | "mul" | "sub" | "div" => {
            // Les opérations arithmétiques nécessitent des types compatibles
            if dtype_a.is_floating() && dtype_b.is_integer() {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Structural,
                    severity: Severity::Warning,
                    message: format!(
                        "Opération {}: mélange de types flottant ({}) et entier ({})",
                        operation, dtype_a, dtype_b
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            } else if dtype_a.is_integer() && dtype_b.is_floating() {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Structural,
                    severity: Severity::Warning,
                    message: format!(
                        "Opération {}: mélange de type entier ({}) et flottant ({})",
                        operation, dtype_a, dtype_b
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            }
        },
        "matmul" | "dot" => {
            // Les opérations matricielles nécessitent des types numériques
            if !dtype_a.is_floating() && !dtype_a.is_integer() {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Structural,
                    severity: Severity::Error,
                    message: format!(
                        "Opération {}: dtype A ({}) non numérique",
                        operation, dtype_a
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            }
            if !dtype_b.is_floating() && !dtype_b.is_integer() {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Structural,
                    severity: Severity::Error,
                    message: format!(
                        "Opération {}: dtype B ({}) non numérique",
                        operation, dtype_b
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            }
        },
        _ => {},
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_matching() {
        let result = validate_dtype("test", SimpleDType::F32, SimpleDType::F32);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn dtype_mismatch() {
        let result = validate_dtype("test", SimpleDType::F32, SimpleDType::F16);
        assert!(!result.issues.is_empty());
        assert!(result.issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn dtype_compatibility_add() {
        let issues =
            validate_dtype_compatibility("test", SimpleDType::F32, SimpleDType::F32, "add");
        assert!(issues.is_empty());
    }

    #[test]
    fn dtype_compatibility_mixed() {
        let issues =
            validate_dtype_compatibility("test", SimpleDType::F32, SimpleDType::I32, "add");
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.severity == Severity::Warning));
    }

    #[test]
    fn dtype_compatibility_matmul() {
        let issues =
            validate_dtype_compatibility("test", SimpleDType::F32, SimpleDType::F32, "matmul");
        assert!(issues.is_empty());
    }
}
