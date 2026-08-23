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

//! Structures de configuration pour la sortie des pseudo-modèles générés.

use std::path::PathBuf;

/// Configuration de la sortie pour un pseudo-modèle généré.
///
/// Contient toutes les informations nécessaires à la création de la
/// structure de dossier de sortie.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Chemin du répertoire de sortie final
    pub output_dir: PathBuf,
    /// Chemin du répertoire source contenant les fichiers de configuration
    pub source_dir: PathBuf,
    /// Modèle source (GLM-5.2 ou DeepSeek-V4-Flash)
    pub source_model: SourceModel,
    /// Seed utilisé pour la génération
    pub seed: u64,
    /// Version du générateur
    pub generator_version: String,
    /// Timestamp UTC de la génération (format ISO 8601)
    pub timestamp_utc: String,
    /// Nombre total de paramètres (0 si inconnu)
    pub parameter_count: u64,
    /// Nombre total de tenseurs
    pub tensor_count: u32,
    /// Nombre de shards
    pub shards: u32,
    /// Taille cible en octets
    pub target_size_bytes: u64,
    /// Taille estimée en octets
    pub estimated_size_bytes: u64,
    /// Taille réelle en octets
    pub actual_size_bytes: u64,
    /// Type de données (bf16, f32, etc.)
    pub dtype: String,
    /// Mode de génération (size-constrained, full-structural)
    pub generation_mode: String,
}

/// Modèle source pour la génération.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceModel {
    /// Modèle GLM-5.2
    Glm52,
    /// Modèle DeepSeek-V4-Flash
    DeepSeekV4Flash,
}

impl SourceModel {
    /// Retourne le nom du modèle source.
    pub fn name(&self) -> &'static str {
        match self {
            SourceModel::Glm52 => "glm-5.2",
            SourceModel::DeepSeekV4Flash => "deepseek-v4-flash",
        }
    }

    /// Retourne la version du profil correspondant au modèle.
    pub fn profile_version(&self) -> &'static str {
        match self {
            SourceModel::Glm52 => "glm52-v1",
            SourceModel::DeepSeekV4Flash => "dsv4f-v1",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test des structures de données.
    #[test]
    fn test_source_model() {
        assert_eq!(SourceModel::Glm52.name(), "glm-5.2");
        assert_eq!(SourceModel::DeepSeekV4Flash.name(), "deepseek-v4-flash");
        assert_eq!(SourceModel::Glm52.profile_version(), "glm52-v1");
        assert_eq!(SourceModel::DeepSeekV4Flash.profile_version(), "dsv4f-v1");
    }
}
