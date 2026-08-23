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

//! Sous-module contenant les implémentations des types de rapport.

use crate::inspector::{InspectionLevel, InspectionReport};

use super::report_formatters::{format_bytes, format_number};
use super::types::{
    ArchitectureSummaryJson, ConfigInspectionJson, PhysicalStatsJson, SafetensorsHeaderJson,
    ShardIndexJson, StructuralStatsJson, StructuredReport,
};

impl StructuredReport {
    /// Crée un rapport structuré à partir d'un rapport d'inspection.
    ///
    /// # Paramètres
    /// - `report` : rapport d'inspection standard.
    pub fn from_inspection_report(report: &InspectionReport) -> Self {
        Self {
            model_path: report.model_path.clone(),
            level: format!("{:?}", report.level),
            timestamp: None,
            config: report.config.as_ref().map(ConfigInspectionJson::from),
            safetensors_headers: report
                .safetensors_headers
                .iter()
                .map(SafetensorsHeaderJson::from)
                .collect(),
            shard_index: report.shard_index.as_ref().map(ShardIndexJson::from),
            structural: StructuralStatsJson::from(&report.structural),
            physical: PhysicalStatsJson::from(&report.physical),
            architecture: ArchitectureSummaryJson::from(&report.architecture),
        }
    }

    /// Ajoute un timestamp au rapport.
    pub fn with_timestamp(mut self, timestamp: String) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Convertit le rapport en JSON pretty-printé.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convertit le rapport en JSON compact.
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Génère un rapport texte selon le niveau de détail.
    pub fn to_text(&self, level: InspectionLevel) -> String {
        match level {
            InspectionLevel::Brief => self.to_brief_text(),
            InspectionLevel::Normal => self.to_normal_text(),
            InspectionLevel::Verbose => self.to_verbose_text(),
            InspectionLevel::Debug => self.to_debug_text(),
        }
    }

    /// Génère un rapport bref en texte.
    fn to_brief_text(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Inspection bref du modèle ===\n");
        output.push_str(&format!("Chemin : {}\n", self.model_path.display()));
        output.push_str(&format!("{}\n", self.architecture));
        output.push_str(&format!(
            "Paramètres : {}\n",
            format_number(self.structural.total_parameters)
        ));
        output.push_str(&format!(
            "Mémoire   : {}\n",
            format_bytes(self.physical.total_memory_bytes)
        ));
        output
    }

    /// Génère un rapport normal en texte.
    fn to_normal_text(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Rapport d'inspection du modèle ===\n");
        output.push_str(&format!("Chemin : {}\n", self.model_path.display()));
        output.push('\n');

        // Configuration
        if let Some(ref config) = self.config {
            output.push_str("--- Configuration ---\n");
            output.push_str(&format!("{}\n", config));
        }

        // Architecture
        output.push('\n');
        output.push_str("--- Architecture ---\n");
        output.push_str(&format!("{}\n", self.architecture));

        // Statistiques structurelles
        output.push('\n');
        output.push_str("--- Statistiques structurelles ---\n");
        output.push_str(&format!("{}\n", self.structural));

        // Statistiques physiques
        output.push('\n');
        output.push_str("--- Statistiques physiques ---\n");
        output.push_str(&format!("{}\n", self.physical));

        output
    }

    /// Génère un rapport verbeux en texte.
    fn to_verbose_text(&self) -> String {
        let mut output = String::new();
        // D'abord le format normal
        output.push_str(&self.to_normal_text());

        // Puis les détails supplémentaires
        output.push('\n');
        output.push_str("--- Détails Safetensors ---\n");
        for (i, header) in self.safetensors_headers.iter().enumerate() {
            output.push_str(&format!(
                "Shard {}: {}\n",
                i + 1,
                header.file_path.display()
            ));
            output.push_str(&format!("  Tenseurs : {}\n", header.tensor_count));
            output.push_str(&format!(
                "  Taille  : {}\n",
                format_bytes(header.total_bytes)
            ));

            // Détails des tenseurs si disponibles
            if let Some(ref tensors) = header.tensors {
                for tensor in tensors.iter().take(5) {
                    output.push_str(&format!(
                        "    {} : {:?} {:?} ({})\n",
                        tensor.name,
                        tensor.shape,
                        tensor.dtype,
                        format_bytes(tensor.size_bytes)
                    ));
                }
                if tensors.len() > 5 {
                    output.push_str(&format!(
                        "    ... et {} autres tenseurs\n",
                        tensors.len() - 5
                    ));
                }
            }
        }

        if let Some(ref index) = self.shard_index {
            output.push('\n');
            output.push_str("--- Index des shards ---\n");
            output.push_str(&format!("{}\n", index));
        }

        output
    }

    /// Génère un rapport debug en texte.
    fn to_debug_text(&self) -> String {
        let mut output = String::new();
        // D'abord le format verbeux
        output.push_str(&self.to_verbose_text());

        // Puis les informations de débogage
        output.push('\n');
        output.push_str("--- Informations de débogage ---\n");
        output.push_str(&format!(
            "Nombre de shards : {}\n",
            self.safetensors_headers.len()
        ));
        output.push_str(&format!(
            "Nombre total de tenseurs : {}\n",
            self.structural.total_tensors
        ));
        output.push_str(&format!(
            "Densité moyenne : {:.4}\n",
            self.physical.average_density
        ));

        // Détails de chaque tenseur (premiers 10)
        if !self.safetensors_headers.is_empty() {
            output.push('\n');
            output.push_str("Premiers tenseurs (max 10) :\n");
            let mut count = 0;
            for header in &self.safetensors_headers {
                if let Some(ref tensors) = header.tensors {
                    for tensor in tensors.iter().take(10 - count) {
                        output.push_str(&format!(
                            "  {} : {:?} {:?} ({})\n",
                            tensor.name,
                            tensor.shape,
                            tensor.dtype,
                            format_bytes(tensor.size_bytes)
                        ));
                        count += 1;
                        if count >= 10 {
                            break;
                        }
                    }
                }
                if count >= 10 {
                    break;
                }
            }
        }

        output
    }
}
