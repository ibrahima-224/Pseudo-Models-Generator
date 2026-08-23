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

//! Configuration étendue du générateur de pseudo-modèles.
//!
//! Ce module définit la structure [`GeneratorConfig`] qui étend [`CoreConfig`]
//! avec les paramètres spécifiques au CLI et au streaming. La composition
//! permet de séparer les types fondamentaux (pmg-core) des paramètres
//! métier (pmg-generator).
//!
//! Conformité : ADR-002, étape 1 - Split GeneratorConfig.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_core::CoreConfig;
//! use pmg_generator::generator_config::GeneratorConfig;
//!
//! let core = CoreConfig::new(42, "glm-5.2").unwrap();
//! let config = GeneratorConfig::from_core(core);
//! assert_eq!(config.core().seed, 42);
//! ```

use std::ops::{Deref, DerefMut};

use pmg_core::core_config::GenerationMode;
use pmg_core::CoreConfig;
use serde::{Deserialize, Serialize};

/// Configuration étendue du générateur.
///
/// Contient le [`CoreConfig`] par composition et ajoute les paramètres
/// spécifiques au CLI, au streaming et au debug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Configuration fondamentale (types purs).
    pub core: CoreConfig,
    /// Mode de génération.
    pub mode: GenerationMode,
    /// Taille des chunks en octets (défaut: 64 Mo).
    pub chunk_size: u64,
    /// Taille maximale par shard en octets (défaut: 5 Go).
    pub max_shard_bytes: u64,
    /// Activer la validation post-génération.
    pub validate: bool,
    /// Mode sec (simuler sans écrire).
    pub dry_run: bool,
    /// Affichage verbeux.
    pub verbose: bool,
    /// Mode debug (très verbeux).
    pub debug: bool,
}

impl GeneratorConfig {
    /// Crée une configuration à partir d'un [`CoreConfig`].
    ///
    /// Les paramètres CLI prennent des valeurs par défaut raisonnables.
    pub fn from_core(core: CoreConfig) -> Self {
        Self {
            core,
            mode: GenerationMode::Safe,
            chunk_size: 67_108_864,         // 64 Mo
            max_shard_bytes: 5_368_709_120, // 5 Go
            validate: true,
            dry_run: false,
            verbose: false,
            debug: false,
        }
    }

    /// Crée une configuration à partir des paramètres CLI.
    ///
    /// # Paramètres
    /// - `seed` : seed de génération ;
    /// - `model_id` : identifiant du modèle ;
    /// - `target_size_bytes` : taille cible en octets ;
    /// - `dtype` : type de données de sortie ;
    /// - `mode` : mode de génération ;
    /// - `chunk_size` : taille des chunks en octets ;
    /// - `max_shard_bytes` : taille maximale par shard ;
    /// - `validate` : activer la validation ;
    /// - `dry_run` : mode sec ;
    /// - `verbose` : affichage verbeux ;
    /// - `debug` : mode debug.
    #[allow(clippy::too_many_arguments)]
    pub fn from_cli_args(
        seed: u64,
        model_id: impl Into<String>,
        target_size_bytes: u64,
        dtype: impl Into<String>,
        mode: GenerationMode,
        chunk_size: u64,
        max_shard_bytes: u64,
        validate: bool,
        dry_run: bool,
        verbose: bool,
        debug: bool,
    ) -> pmg_core::CoreResult<Self> {
        let mut core = CoreConfig::new(seed, model_id)?;
        core.target_size_bytes = target_size_bytes;
        core.dtype = dtype.into();

        Ok(Self {
            core,
            mode,
            chunk_size,
            max_shard_bytes,
            validate,
            dry_run,
            verbose,
            debug,
        })
    }

    /// Retourne une référence au [`CoreConfig`].
    pub fn core(&self) -> &CoreConfig {
        &self.core
    }

    /// Retourne une référence mutable au [`CoreConfig`].
    pub fn core_mut(&mut self) -> &mut CoreConfig {
        &mut self.core
    }

    /// Valide la cohérence interne de la configuration.
    pub fn validate(&self) -> pmg_core::CoreResult<()> {
        // Valide le CoreConfig
        self.core.validate()?;

        // Valide les paramètres spécifiques
        if self.chunk_size == 0 {
            return Err(pmg_core::CoreError::Validation(
                "chunk_size ne peut pas être 0".into(),
            ));
        }
        if self.max_shard_bytes == 0 {
            return Err(pmg_core::CoreError::Validation(
                "max_shard_bytes ne peut pas être 0".into(),
            ));
        }
        Ok(())
    }

    /// Crée une configuration à partir d'un profil statistique.
    ///
    /// # Paramètres
    /// - `seed` : seed de génération ;
    /// - `model_id` : identifiant du modèle ;
    /// - `profile` : profil statistique à utiliser.
    pub fn from_statistical_profile(
        seed: u64,
        model_id: impl Into<String>,
        profile: &pmg_core::StatisticalProfile,
    ) -> pmg_core::CoreResult<Self> {
        let core = CoreConfig::from_statistical_profile(seed, model_id, profile)?;
        Ok(Self::from_core(core))
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self::from_core(CoreConfig::default())
    }
}

/// Implémentation de `Deref` pour permettre l'accès transparent aux champs de `CoreConfig`.
impl Deref for GeneratorConfig {
    type Target = CoreConfig;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

/// Implémentation de `DerefMut` pour permettre la modification transparente des champs de `CoreConfig`.
impl DerefMut for GeneratorConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_config_creation() {
        let core = CoreConfig::new(42, "glm-5.2").unwrap();
        let config = GeneratorConfig::from_core(core);
        assert_eq!(config.core().seed, 42);
        assert_eq!(config.core().model_id, "glm-5.2");
        assert_eq!(config.mode, GenerationMode::Safe);
        assert_eq!(config.chunk_size, 67_108_864);
    }

    #[test]
    fn generator_config_from_cli_args() {
        let config = GeneratorConfig::from_cli_args(
            42,
            "glm-5.2",
            1_000_000_000,
            "f16",
            GenerationMode::Realistic,
            134_217_728,
            10_737_418_240,
            false,
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(config.seed, 42);
        assert_eq!(config.model_id, "glm-5.2");
        assert_eq!(config.target_size_bytes, 1_000_000_000);
        assert_eq!(config.dtype, "f16");
        assert_eq!(config.mode, GenerationMode::Realistic);
        assert_eq!(config.chunk_size, 134_217_728);
        assert_eq!(config.max_shard_bytes, 10_737_418_240);
        assert!(!config.validate);
        assert!(config.dry_run);
        assert!(config.verbose);
        assert!(!config.debug);
    }

    #[test]
    fn generator_config_validation() {
        let mut config = GeneratorConfig::default();
        assert!(config.validate().is_ok());

        // Invalide : chunk_size = 0
        config.chunk_size = 0;
        assert!(config.validate().is_err());

        // Rétablit et teste max_shard_bytes
        config.chunk_size = 67_108_864;
        config.max_shard_bytes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn generator_config_deref() {
        let core = CoreConfig::new(42, "test").unwrap();
        let config = GeneratorConfig::from_core(core);

        // Accès transparent via Deref
        assert_eq!(config.seed, 42);
        assert_eq!(config.model_id, "test");
    }

    #[test]
    fn generator_config_deref_mut() {
        // Utilisation de la syntaxe struct update pour éviter clippy::field_reassign_with_default
        let config = GeneratorConfig {
            core: CoreConfig {
                seed: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.core().seed, 100);
    }
}
