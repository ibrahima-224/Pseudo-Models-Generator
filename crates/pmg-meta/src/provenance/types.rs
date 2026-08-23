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

//! Sous-module contenant les types de base pour la provenance.

use serde::{Deserialize, Serialize};

/// Informations sur une source de métadonnées.
///
/// Représente une source de données utilisée comme entrée pour la génération.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceMetadata {
    /// Chemin ou identifiant de la source.
    pub path: String,
    /// Hash SHA-256 du contenu.
    pub hash: String,
    /// Taille en octets.
    pub size_bytes: u64,
    /// Horodatage de la dernière modification.
    pub last_modified: Option<String>,
    /// Type de source (ex: "config.json", "tokenizer.json").
    pub source_type: String,
    /// Version de la source (si disponible).
    pub version: Option<String>,
}

/// Informations sur les métadonnées d'entrée utilisées.
///
/// Regroupe toutes les sources de données utilisées pour la génération.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputMetadata {
    /// Métadonnées du modèle source.
    pub model_config: Option<SourceMetadata>,
    /// Métadonnées du tokenizer.
    pub tokenizer_config: Option<SourceMetadata>,
    /// Métadonnées du profil statistique.
    pub statistical_profile: Option<SourceMetadata>,
    /// Métadonnées du blueprint.
    pub blueprint: Option<SourceMetadata>,
    /// Autres sources utilisées.
    pub additional_sources: Vec<SourceMetadata>,
}

/// Artifact généré lors de la génération.
///
/// Représente un fichier produit par le processus de génération.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeneratedArtifact {
    /// Chemin de l'artifact.
    pub path: String,
    /// Type d'artifact (ex: "manifest", "statistics", "weights").
    pub artifact_type: String,
    /// Taille en octets.
    pub size_bytes: u64,
    /// Hash SHA-256 du contenu.
    pub hash: String,
    /// Horodatage de création.
    pub created_at: String,
}

/// Environnement de génération.
///
/// Décrit les caractéristiques du système utilisé pour la génération.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationEnvironment {
    /// Système d'exploitation.
    pub os: String,
    /// Architecture CPU.
    pub arch: String,
    /// Version de Rust utilisée.
    pub rust_version: String,
    /// Mémoire totale disponible (en octets).
    pub total_memory_bytes: u64,
    /// Nombre de cœurs CPU.
    pub cpu_cores: u32,
    /// Horodatage de début de génération.
    pub start_time: String,
    /// Horodatage de fin de génération.
    pub end_time: String,
}
