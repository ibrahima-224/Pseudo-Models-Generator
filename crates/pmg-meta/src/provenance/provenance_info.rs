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

//! Sous-module contenant la structure principale de provenance.

use serde::{Deserialize, Serialize};

use super::types::{GeneratedArtifact, GenerationEnvironment, InputMetadata};

/// Informations de provenance pour la génération.
///
/// Contient toutes les informations de traçabilité pour une génération PMG.
///
/// # Validations
///
/// La validation vérifie :
/// - `generation_id` n'est pas vide
/// - `timestamp_utc` n'est pas vide
/// - `pmg_version` n'est pas vide
/// - Au moins un artifact est généré
/// - Les timestamps sont cohérents (start <= end)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceInfo {
    /// Version du schéma de provenance.
    pub schema_version: u32,
    /// Identifiant unique de la génération.
    pub generation_id: String,
    /// Horodatage UTC de la génération.
    pub timestamp_utc: String,
    /// Version du générateur PMG.
    pub pmg_version: String,
    /// Version du générateur spécifique.
    pub generator_version: String,
    /// Graine aléatoire utilisée.
    pub seed: u64,
    /// Mode de génération.
    pub generation_mode: String,
    /// Métadonnées d'entrée utilisées.
    pub input_metadata: InputMetadata,
    /// Sorties générées.
    pub generated_artifacts: Vec<GeneratedArtifact>,
    /// Environnement de génération.
    pub environment: GenerationEnvironment,
}

impl ProvenanceInfo {
    /// Crée une nouvelle instance de provenance avec des valeurs par défaut.
    pub fn new(generation_id: &str, seed: u64, generation_mode: &str) -> Self {
        Self {
            schema_version: 1,
            generation_id: generation_id.to_string(),
            timestamp_utc: chrono::Utc::now().to_rfc3339(),
            pmg_version: "1.0.0".to_string(),
            generator_version: "1.0.0".to_string(),
            seed,
            generation_mode: generation_mode.to_string(),
            input_metadata: InputMetadata {
                model_config: None,
                tokenizer_config: None,
                statistical_profile: None,
                blueprint: None,
                additional_sources: Vec::new(),
            },
            generated_artifacts: Vec::new(),
            environment: GenerationEnvironment {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
                total_memory_bytes: 0,
                cpu_cores: num_cpus::get() as u32,
                start_time: String::new(),
                end_time: String::new(),
            },
        }
    }

    /// Ajoute un artifact généré.
    pub fn add_artifact(&mut self, artifact: GeneratedArtifact) {
        self.generated_artifacts.push(artifact);
    }

    /// Définit les métadonnées d'entrée.
    pub fn set_input_metadata(&mut self, metadata: InputMetadata) {
        self.input_metadata = metadata;
    }

    /// Définit l'environnement de génération.
    pub fn set_environment(&mut self, env: GenerationEnvironment) {
        self.environment = env;
    }

    /// Calcule le hash de traçabilité complet.
    ///
    /// Génère un hash SHA-256 unique basé sur les paramètres de génération
    /// pour permettre l'identification et la vérification d'intégrité.
    pub fn traceability_hash(&self) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();

        // Ajout des données de base
        hasher.update(self.generation_id.as_bytes());
        hasher.update(self.seed.to_le_bytes());
        hasher.update(self.generation_mode.as_bytes());
        hasher.update(self.timestamp_utc.as_bytes());

        // Ajout des hashes des métadonnées d'entrée
        if let Some(ref config) = self.input_metadata.model_config {
            hasher.update(config.hash.as_bytes());
        }
        if let Some(ref tokenizer) = self.input_metadata.tokenizer_config {
            hasher.update(tokenizer.hash.as_bytes());
        }

        let result = hasher.finalize();
        // Formater le hash en hexadécimal manuellement (sha2 0.11 ne satisfait pas LowerHex)
        let hex_str: String = result.iter().map(|b| format!("{:02x}", b)).collect();
        format!("sha256:{}", hex_str)
    }

    /// Vérifie la cohérence de la provenance.
    pub fn validate(&self) -> Result<(), String> {
        if self.generation_id.is_empty() {
            return Err("L'identifiant de génération ne peut pas être vide".to_string());
        }

        if self.timestamp_utc.is_empty() {
            return Err("L'horodatage ne peut pas être vide".to_string());
        }

        if self.pmg_version.is_empty() {
            return Err("La version PMG ne peut pas être vide".to_string());
        }

        if self.generated_artifacts.is_empty() {
            return Err("Aucun artifact généré trouvé".to_string());
        }

        // Vérification des timestamps
        if self.environment.start_time > self.environment.end_time
            && !self.environment.start_time.is_empty()
            && !self.environment.end_time.is_empty()
        {
            return Err("L'horodatage de fin ne peut pas précéder le début".to_string());
        }

        Ok(())
    }

    /// Retourne un résumé textuel de la provenance.
    pub fn summary(&self) -> String {
        format!(
            "Provenance pour la génération {}\n\
             PMG v{}, graine: {}, mode: {}\n\
             Artifacts générés: {}, sources d'entrée: {}\n\
             Environnement: {} {} (Rust {})\n\
             Mémoire: {} octets, CPU: {} cœurs",
            self.generation_id,
            self.pmg_version,
            self.seed,
            self.generation_mode,
            self.generated_artifacts.len(),
            self.count_input_sources(),
            self.environment.os,
            self.environment.arch,
            self.environment.rust_version,
            self.environment.total_memory_bytes,
            self.environment.cpu_cores
        )
    }

    /// Compte le nombre de sources d'entrée.
    fn count_input_sources(&self) -> usize {
        let mut count = 0;
        if self.input_metadata.model_config.is_some() {
            count += 1;
        }
        if self.input_metadata.tokenizer_config.is_some() {
            count += 1;
        }
        if self.input_metadata.statistical_profile.is_some() {
            count += 1;
        }
        if self.input_metadata.blueprint.is_some() {
            count += 1;
        }
        count += self.input_metadata.additional_sources.len();
        count
    }
}

impl std::fmt::Display for ProvenanceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}
