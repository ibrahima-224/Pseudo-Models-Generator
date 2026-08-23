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

//! Fonctions utilitaires pour la comparaison architecturale.
//!
//! Ce module contient des helpers pour l'extraction de valeurs,
//! la comparaison d'options et la génération de diffs.

use crate::config_compare::ConfigValue;
use crate::diff::Diff;

/// Extrait une valeur i64 d'un ConfigValue.
///
/// # Arguments
///
/// * `value` - Le ConfigValue à convertir.
///
/// # Retour
///
/// La valeur i64 si la conversion est possible, sinon None.
pub(crate) fn extract_i64(value: &ConfigValue) -> Option<i64> {
    match value {
        ConfigValue::Integer(i) => Some(*i),
        ConfigValue::Float(f) => Some(*f as i64),
        _ => None,
    }
}

/// Compare deux options de i64 et génère un Diff si nécessaire.
///
/// # Arguments
///
/// * `name` - Nom de la propriété comparée.
/// * `a` - Première valeur optionnelle.
/// * `b` - Deuxième valeur optionnelle.
/// * `description_missing` - Description si une valeur est manquante.
/// * `description_different` - Description si les valeurs sont différentes.
///
/// # Retour
///
/// Un tuple (compatible, diffs) où compatible est true si les valeurs sont égales.
pub(crate) fn compare_optional_i64(
    name: &str,
    a: Option<i64>,
    b: Option<i64>,
    description_missing: &str,
    description_different: &str,
) -> (bool, Option<Diff>) {
    match (a, b) {
        (Some(a_val), Some(b_val)) => {
            if a_val == b_val {
                (true, None)
            } else {
                (
                    false,
                    Some(Diff::modified(
                        name.to_string(),
                        a_val.to_string(),
                        b_val.to_string(),
                        description_different.to_string(),
                    )),
                )
            }
        },
        _ => (
            false,
            Some(Diff::modified(
                name.to_string(),
                a.map(|v| v.to_string())
                    .unwrap_or_else(|| "N/A".to_string()),
                b.map(|v| v.to_string())
                    .unwrap_or_else(|| "N/A".to_string()),
                description_missing.to_string(),
            )),
        ),
    }
}

/// Compare deux options de i64 sans générer de Diff si une valeur est manquante.
///
/// Cette fonction est utilisée pour les propriétés non bloquantes comme num_experts.
///
/// # Arguments
///
/// * `name` - Nom de la propriété comparée.
/// * `a` - Première valeur optionnelle.
/// * `b` - Deuxième valeur optionnelle.
/// * `description_different` - Description si les valeurs sont différentes.
///
/// # Retour
///
/// Un tuple (compatible, diffs) où compatible est true si les valeurs sont égales.
pub(crate) fn compare_optional_i64_non_blocking(
    name: &str,
    a: Option<i64>,
    b: Option<i64>,
    description_different: &str,
) -> (bool, Option<Diff>) {
    match (a, b) {
        (Some(a_val), Some(b_val)) => {
            if a_val == b_val {
                (true, None)
            } else {
                (
                    false,
                    Some(Diff::modified(
                        name.to_string(),
                        a_val.to_string(),
                        b_val.to_string(),
                        description_different.to_string(),
                    )),
                )
            }
        },
        _ => {
            // Si au moins une valeur est présente et elles sont différentes
            if a.is_some() || b.is_some() {
                (
                    false,
                    Some(Diff::modified(
                        name.to_string(),
                        a.map(|v| v.to_string())
                            .unwrap_or_else(|| "N/A".to_string()),
                        b.map(|v| v.to_string())
                            .unwrap_or_else(|| "N/A".to_string()),
                        format!("{} (non bloquant)", description_different),
                    )),
                )
            } else {
                // Les deux sont absentes, considéré comme compatible
                (true, None)
            }
        },
    }
}
