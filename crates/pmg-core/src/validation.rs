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

//! Fonctions de validation partagées de la crate `pmg-core`.
//!
//! Ces prédicats centralisent les invariants transverses réutilisés par
//! `Shape`, `TensorMetadata`, `ModelConfig` et le planner :
//! tailles strictement positives, offsets contigus, divisibilité.

use crate::error::{CoreError, CoreResult};

/// Vérifie qu'une taille est strictement positive.
///
/// # Erreurs
/// - [`CoreError::InvalidShape`] si `size == 0` (message français explicite).
pub fn validate_non_zero_size(size: u64, what: &str) -> CoreResult<()> {
    if size == 0 {
        return Err(CoreError::invalid_shape(format!(
            "{what} doit être strictement positif (valeur 0 interdite)"
        )));
    }
    Ok(())
}

/// Vérifie la cohérence d'un intervalle `[offset_start, offset_end)`.
///
/// Règles :
/// - `offset_start <= offset_end` (intervalle non inversé) ;
/// - `offset_end - offset_start == expected_len` (contiguïté exacte) ;
/// - `expected_len > 0` sauf si `allow_empty` est vrai.
pub fn validate_offset_range(
    offset_start: u64,
    offset_end: u64,
    expected_len: u64,
    allow_empty: bool,
) -> CoreResult<()> {
    if offset_start > offset_end {
        return Err(CoreError::Validation(format!(
            "offset de début ({offset_start}) supérieur à l'offset de fin ({offset_end})"
        )));
    }
    let len = offset_end
        .checked_sub(offset_start)
        .ok_or_else(|| CoreError::Overflow("soustraction d'offsets".into()))?;
    if len != expected_len {
        return Err(CoreError::Validation(format!(
            "longueur d'intervalle incohérente : attendu {expected_len}, obtenu {len} \
             (offsets [{offset_start}, {offset_end})"
        )));
    }
    if len == 0 && !allow_empty {
        return Err(CoreError::Validation(
            "intervalle vide interdit pour cette entité".into(),
        ));
    }
    Ok(())
}

/// Vérifie que `hidden_size` est divisible par `num_heads`.
///
/// # Erreurs
/// - [`CoreError::IncompatibleHeads`] si la division n'est pas exacte ou si
///   `num_heads == 0`.
pub fn validate_divisible_by(hidden_size: u64, num_heads: u64, what: &str) -> CoreResult<u64> {
    if num_heads == 0 {
        return Err(CoreError::IncompatibleHeads(format!(
            "le nombre de têtes de {what} ne peut pas être nul"
        )));
    }
    if hidden_size % num_heads != 0 {
        return Err(CoreError::IncompatibleHeads(format!(
            "{what}: hidden_size {hidden_size} n'est pas divisible par {num_heads} têtes"
        )));
    }
    Ok(hidden_size / num_heads)
}

/// Vérifie que `n >= 1` (nombre de couches, vocabulaire, experts…).
///
/// # Erreurs
/// - [`CoreError::InvalidModelConfig`] si `n == 0`.
pub fn validate_at_least_one(n: u64, what: &str) -> CoreResult<()> {
    if n == 0 {
        return Err(CoreError::InvalidModelConfig(format!(
            "{what} doit être ≥ 1 (valeur 0 interdite)"
        )));
    }
    Ok(())
}

/// Vérifie qu'une valeur strictement positive (`rope_theta`, `rms_norm_eps`…).
pub fn validate_strictly_positive(value: f64, what: &str) -> CoreResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CoreError::InvalidModelConfig(format!(
            "{what} doit être fini et strictement positif (obtenu {value})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        validate_at_least_one, validate_divisible_by, validate_non_zero_size,
        validate_offset_range, validate_strictly_positive,
    };
    use crate::error::CoreError;

    #[test]
    fn non_zero_size() {
        assert!(validate_non_zero_size(1, "hidden_size").is_ok());
        assert!(matches!(
            validate_non_zero_size(0, "hidden_size"),
            Err(CoreError::InvalidShape(_))
        ));
    }

    #[test]
    fn offset_range_validation() {
        // Intervalle contigu valide.
        assert!(validate_offset_range(0, 8, 8, false).is_ok());
        // Intervalle inversé.
        assert!(matches!(
            validate_offset_range(8, 4, 4, false),
            Err(CoreError::Validation(_))
        ));
        // Longueur incohérente.
        assert!(matches!(
            validate_offset_range(0, 10, 8, false),
            Err(CoreError::Validation(_))
        ));
        // Intervalle vide interdit par défaut, toléré si allow_empty.
        assert!(validate_offset_range(4, 4, 0, true).is_ok());
        assert!(matches!(
            validate_offset_range(4, 4, 0, false),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn divisibility() {
        assert_eq!(validate_divisible_by(6144, 64, "GLM").unwrap(), 96);
        assert!(matches!(
            validate_divisible_by(6144, 0, "x"),
            Err(CoreError::IncompatibleHeads(_))
        ));
        assert!(matches!(
            validate_divisible_by(10, 3, "x"),
            Err(CoreError::IncompatibleHeads(_))
        ));
    }

    #[test]
    fn at_least_one() {
        assert!(validate_at_least_one(78, "num_layers").is_ok());
        assert!(matches!(
            validate_at_least_one(0, "num_layers"),
            Err(CoreError::InvalidModelConfig(_))
        ));
    }

    #[test]
    fn strictly_positive() {
        assert!(validate_strictly_positive(1e-5, "rms_norm_eps").is_ok());
        assert!(validate_strictly_positive(0.0, "rms_norm_eps").is_err());
        assert!(validate_strictly_positive(f64::NAN, "x").is_err());
        assert!(validate_strictly_positive(f64::INFINITY, "x").is_err());
    }
}
