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

//! Module principal d'inspection des modèles.
//!
//! Fournit la structure [`ModelInspector`] qui coordonne les différentes sources
//! de métadonnées (configuration, headers Safetensors, index, statistiques)
//! sans jamais charger les poids complets (principe Zero-Payload).
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::inspector::ModelInspector;
//!
//! // Création d'un inspecteur pour un modèle (chemin fictif)
//! // let inspector = ModelInspector::new("path/to/model");
//! // let report = inspector.inspect().unwrap();
//! // println!("{}", report);
//! ```

use std::path::{Path, PathBuf};

use crate::architecture::ArchitectureSummary;
use crate::config_inspector::ConfigInspection;
use crate::index_inspector::ShardIndex;
use crate::physical_stats::PhysicalStats;
use crate::safetensors_inspector::SafetensorsHeader;
use crate::structural_stats::StructuralStats;

/// Niveau de détail du rapport d'inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InspectionLevel {
    /// Affichage bref (résumé).
    Brief,
    /// Affichage normal (détails principaux).
    #[default]
    Normal,
    /// Affichage verbeux (tous les détails).
    Verbose,
    /// Affichage debug (informations techniques).
    Debug,
}

/// Rapport complet d'inspection d'un modèle.
#[derive(Debug, Clone)]
pub struct InspectionReport {
    /// Chemin du modèle inspecté.
    pub model_path: PathBuf,
    /// Niveau de détail du rapport.
    pub level: InspectionLevel,
    /// Inspection de la configuration.
    pub config: Option<ConfigInspection>,
    /// Headers Safetensors extraits.
    pub safetensors_headers: Vec<SafetensorsHeader>,
    /// Index des shards (si available).
    pub shard_index: Option<ShardIndex>,
    /// Statistiques structurelles.
    pub structural: StructuralStats,
    /// Statistiques physiques.
    pub physical: PhysicalStats,
    /// Résumé architectural.
    pub architecture: ArchitectureSummary,
}

impl InspectionReport {
    /// Crée un nouveau rapport d'inspection vide.
    pub fn new(model_path: PathBuf, level: InspectionLevel) -> Self {
        Self {
            model_path,
            level,
            config: None,
            safetensors_headers: Vec::new(),
            shard_index: None,
            structural: StructuralStats::default(),
            physical: PhysicalStats::default(),
            architecture: ArchitectureSummary::default(),
        }
    }
}

impl std::fmt::Display for InspectionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.level {
            InspectionLevel::Brief => write_brief(self, f),
            InspectionLevel::Normal => write_normal(self, f),
            InspectionLevel::Verbose => write_verbose(self, f),
            InspectionLevel::Debug => write_debug(self, f),
        }
    }
}

/// Structure principale qui coordonne l'inspection d'un modèle.
pub struct ModelInspector {
    /// Chemin vers le répertoire du modèle.
    model_path: PathBuf,
    /// Niveau de détail souhaité.
    pub level: InspectionLevel,
}

impl ModelInspector {
    /// Crée un nouvel inspecteur pour le modèle situé au chemin donné.
    ///
    /// # Paramètres
    /// - `model_path` : chemin vers le répertoire contenant le modèle.
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            level: InspectionLevel::default(),
        }
    }

    /// Définit le niveau de détail du rapport.
    pub fn with_level(mut self, level: InspectionLevel) -> Self {
        self.level = level;
        self
    }

    /// Effectue l'inspection complète du modèle.
    ///
    /// # Erreurs
    /// Retourne une erreur si des fichiers essentiels sont manquants ou illisibles.
    pub fn inspect(&self) -> Result<InspectionReport, crate::InspectError> {
        let mut report = InspectionReport::new(self.model_path.clone(), self.level);

        // 1. Inspection de la configuration
        report.config = Some(crate::config_inspector::inspect_config(&self.model_path)?);

        // 2. Inspection des headers Safetensors
        report.safetensors_headers =
            crate::safetensors_inspector::inspect_safetensors_headers(&self.model_path)?;

        // 3. Construction de l'index des shards
        report.shard_index = Some(crate::index_inspector::build_shard_index(
            &self.model_path,
            &report.safetensors_headers,
        )?);

        // 4. Calcul des statistiques structurelles
        report.structural = crate::structural_stats::compute_structural_stats(
            &report.config,
            &report.safetensors_headers,
            &report.shard_index,
        );

        // 5. Calcul des statistiques physiques
        report.physical = crate::physical_stats::compute_physical_stats(
            &report.safetensors_headers,
            &report.structural,
        );

        // 6. Génération du résumé architectural
        report.architecture =
            crate::architecture::summarize_architecture(&report.config, &report.structural);

        Ok(report)
    }

    /// Retourne le chemin du modèle inspecté.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

// ============================================================================
// Fonctions de formatage selon le niveau de détail
// ============================================================================

fn write_brief(report: &InspectionReport, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "=== Inspection bref du modèle ===")?;
    writeln!(f, "Chemin : {}", report.model_path.display())?;
    writeln!(f, "{}", report.architecture)?;
    writeln!(
        f,
        "Paramètres : {}",
        format_number(report.structural.total_parameters)
    )?;
    writeln!(
        f,
        "Mémoire   : {}",
        format_bytes(report.physical.total_memory_bytes)
    )?;
    Ok(())
}

fn write_normal(report: &InspectionReport, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "=== Rapport d'inspection du modèle ===")?;
    writeln!(f, "Chemin : {}", report.model_path.display())?;
    writeln!(f)?;

    // Configuration
    if let Some(ref config) = report.config {
        writeln!(f, "--- Configuration ---")?;
        writeln!(f, "{}", config)?;
    }

    // Architecture
    writeln!(f)?;
    writeln!(f, "--- Architecture ---")?;
    writeln!(f, "{}", report.architecture)?;

    // Statistiques structurelles
    writeln!(f)?;
    writeln!(f, "--- Statistiques structurelles ---")?;
    writeln!(f, "{}", report.structural)?;

    // Statistiques physiques
    writeln!(f)?;
    writeln!(f, "--- Statistiques physiques ---")?;
    writeln!(f, "{}", report.physical)?;

    Ok(())
}

fn write_verbose(report: &InspectionReport, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // D'abord le format normal
    write_normal(report, f)?;

    // Puis les détails supplémentaires
    writeln!(f)?;
    writeln!(f, "--- Détails Safetensors ---")?;
    for (i, header) in report.safetensors_headers.iter().enumerate() {
        writeln!(f, "Shard {}: {}", i + 1, header.file_path.display())?;
        writeln!(f, "  Tenseurs : {}", header.tensor_count())?;
        writeln!(f, "  Taille  : {}", format_bytes(header.total_bytes()))?;
    }

    if let Some(ref index) = report.shard_index {
        writeln!(f)?;
        writeln!(f, "--- Index des shards ---")?;
        writeln!(f, "{}", index)?;
    }

    Ok(())
}

fn write_debug(report: &InspectionReport, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // D'abord le format verbeux
    write_verbose(report, f)?;

    // Puis les informations de débogage
    writeln!(f)?;
    writeln!(f, "--- Informations de débogage ---")?;
    writeln!(f, "Nombre de shards : {}", report.safetensors_headers.len())?;
    writeln!(
        f,
        "Nombre total de tenseurs : {}",
        report.structural.total_tensors
    )?;
    writeln!(
        f,
        "Densité moyenne : {:.4}",
        report.physical.average_density
    )?;

    // Détails de chaque tenseur (premiers 10)
    if !report.safetensors_headers.is_empty() {
        writeln!(f)?;
        writeln!(f, "Premiers tenseurs (max 10) :")?;
        let mut count = 0;
        for header in &report.safetensors_headers {
            for tensor in header.tensors.iter().take(10 - count) {
                writeln!(
                    f,
                    "  {} : {:?} {:?} ({})",
                    tensor.name,
                    tensor.shape,
                    tensor.dtype,
                    format_bytes(tensor.size_bytes())
                )?;
                count += 1;
                if count >= 10 {
                    break;
                }
            }
            if count >= 10 {
                break;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Fonctions utilitaires de formatage
// ============================================================================

/// Formate un nombre avec des séparateurs de milliers.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Formate une taille en octets en unité lisible.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TiB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{} o", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1_234");
        assert_eq!(format_number(1234567), "1_234_567");
        assert_eq!(format_number(1234567890), "1_234_567_890");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 o");
        assert_eq!(format_bytes(1023), "1023 o");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1048576), "1.00 MiB");
        assert_eq!(format_bytes(1073741824), "1.00 GiB");
        assert_eq!(format_bytes(1099511627776), "1.00 TiB");
    }

    #[test]
    fn test_inspection_level_default() {
        assert_eq!(InspectionLevel::default(), InspectionLevel::Normal);
    }

    #[test]
    fn test_model_inspector_creation() {
        let inspector = ModelInspector::new("/path/to/model");
        assert_eq!(inspector.model_path(), Path::new("/path/to/model"));
        assert_eq!(inspector.level, InspectionLevel::Normal);
    }

    #[test]
    fn test_model_inspector_with_level() {
        let inspector = ModelInspector::new("/path/to/model").with_level(InspectionLevel::Verbose);
        assert_eq!(inspector.level, InspectionLevel::Verbose);
    }
}
