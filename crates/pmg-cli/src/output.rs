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

//! Module de gestion centralisée de la sortie pour la CLI PMG.
//!
//! Ce module fournit des fonctions pour afficher des messages utilisateur
//! de manière cohérente et structurée. Il gère les différents niveaux
//! de sortie (succès, erreur, avertissement, info, debug) et assure
//! une expérience utilisateur claire et professionnelle.
//!
//! # Niveaux de sortie
//!
//! | Niveau | Description | Couleur |
//! |--------|-------------|---------|
//! | Succès | Opération terminée avec succès | Vert |
//! | Erreur | Erreur utilisateur ou technique | Rouge |
//! | Avertissement | Attention, résultat potentiellement inattendu | Jaune |
//! | Info | Information générale | Bleu |
//! | Debug | Information de débogage (uniquement avec --debug) | Gris |
//!
//! # Format
//!
//! Tous les messages suivent le format :
//! ```text
//! [NIVEAU] Message détaillé
//! ```
//!
//! Les erreurs incluent toujours un conseil correctif.

use std::fmt;

use crate::exit_codes;

/// Niveaux de sortie disponibles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputLevel {
    /// Message de succès (vert).
    Success,
    /// Message d'erreur (rouge).
    Error,
    /// Message d'avertissement (jaune).
    Warning,
    /// Message d'information (bleu).
    Info,
    /// Message de débogage (gris, uniquement avec --debug).
    Debug,
}

impl OutputLevel {
    /// Retourne le préfixe du niveau en majuscules.
    pub fn prefix(&self) -> &'static str {
        match self {
            OutputLevel::Success => "SUCCÈS",
            OutputLevel::Error => "ERREUR",
            OutputLevel::Warning => "ATTENTION",
            OutputLevel::Info => "INFO",
            OutputLevel::Debug => "DEBUG",
        }
    }

    /// Indique si le message doit être affiché sur stderr.
    pub fn uses_stderr(&self) -> bool {
        matches!(self, OutputLevel::Error | OutputLevel::Warning)
    }
}

/// Structure représentant un message de sortie formaté.
#[derive(Debug, Clone)]
pub struct OutputMessage {
    /// Niveau du message.
    pub level: OutputLevel,
    /// Texte principal du message.
    pub message: String,
    /// Conseil correctif (optionnel, principalement pour les erreurs).
    pub advice: Option<String>,
    /// Cause technique (optionnel, pour les erreurs détaillées).
    pub cause: Option<String>,
}

impl OutputMessage {
    /// Crée un nouveau message de sortie.
    pub fn new(level: OutputLevel, message: &str) -> Self {
        Self {
            level,
            message: message.to_string(),
            advice: None,
            cause: None,
        }
    }

    /// Ajoute un conseil correctif au message.
    pub fn with_advice(mut self, advice: &str) -> Self {
        self.advice = Some(advice.to_string());
        self
    }

    /// Ajoute une cause technique au message.
    pub fn with_cause(mut self, cause: &str) -> Self {
        self.cause = Some(cause.to_string());
        self
    }

    /// Affiche le message sur la sortie appropriée (stdout ou stderr).
    pub fn display(&self) {
        let output = self.to_string();
        if self.level.uses_stderr() {
            eprintln!("{}", output);
        } else {
            println!("{}", output);
        }
    }
}

impl fmt::Display for OutputMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.level.prefix(), self.message)?;

        if let Some(cause) = &self.cause {
            write!(f, "\n  Cause : {}", cause)?;
        }

        if let Some(advice) = &self.advice {
            write!(f, "\n  Conseil : {}", advice)?;
        }

        Ok(())
    }
}

/// Affiche un message de succès.
pub fn success(message: &str) {
    OutputMessage::new(OutputLevel::Success, message).display();
}

/// Affiche un message d'erreur avec un conseil (code 2 : argument invalide).
pub fn error_with_advice(message: &str, advice: &str) {
    let formatted = format!("Erreur PMG-{}: {}", exit_codes::INVALID_ARGUMENT, message);
    OutputMessage::new(OutputLevel::Error, &formatted)
        .with_advice(advice)
        .display();
}

/// Affiche un message d'erreur avec une cause et un conseil (code 2 : argument invalide).
pub fn error_with_cause_and_advice(message: &str, cause: &str, advice: &str) {
    let formatted = format!("Erreur PMG-{}: {}", exit_codes::INVALID_ARGUMENT, message);
    OutputMessage::new(OutputLevel::Error, &formatted)
        .with_cause(cause)
        .with_advice(advice)
        .display();
}

/// Affiche un message d'erreur pour modèle invalide (code 3).
pub fn error_invalid_model(message: &str, cause: &str, advice: &str) {
    let formatted = format!("Erreur PMG-{}: {}", exit_codes::INVALID_MODEL, message);
    OutputMessage::new(OutputLevel::Error, &formatted)
        .with_cause(cause)
        .with_advice(advice)
        .display();
}

/// Affiche un message d'erreur pour erreur I/O (code 4).
pub fn error_io(message: &str, cause: &str, advice: &str) {
    let formatted = format!("Erreur PMG-{}: {}", exit_codes::IO_ERROR, message);
    OutputMessage::new(OutputLevel::Error, &formatted)
        .with_cause(cause)
        .with_advice(advice)
        .display();
}

/// Affiche un message d'erreur pour validation échouée (code 5).
pub fn error_validation(message: &str, cause: &str, advice: &str) {
    let formatted = format!("Erreur PMG-{}: {}", exit_codes::VALIDATION_FAILED, message);
    OutputMessage::new(OutputLevel::Error, &formatted)
        .with_cause(cause)
        .with_advice(advice)
        .display();
}

/// Affiche un message d'erreur pour comparaison incompatible (code 6).
pub fn error_comparison(message: &str, cause: &str, advice: &str) {
    let formatted = format!(
        "Erreur PMG-{}: {}",
        exit_codes::INCOMPATIBLE_COMPARISON,
        message
    );
    OutputMessage::new(OutputLevel::Error, &formatted)
        .with_cause(cause)
        .with_advice(advice)
        .display();
}

/// Affiche un message d'avertissement.
pub fn warning(message: &str) {
    OutputMessage::new(OutputLevel::Warning, message).display();
}

/// Affiche un message d'information.
pub fn info(message: &str) {
    OutputMessage::new(OutputLevel::Info, message).display();
}

/// Affiche un message de débogage (uniquement si debug est activé).
pub fn debug(message: &str, debug_enabled: bool) {
    if debug_enabled {
        OutputMessage::new(OutputLevel::Debug, message).display();
    }
}

/// Affiche un titre de section.
pub fn section(title: &str) {
    println!();
    println!("=== {} ===", title);
}

/// Affiche un sous-titre.
pub fn subsection(title: &str) {
    println!();
    println!("--- {} ---", title);
}

/// Affiche une paire clé-valeur formatée.
pub fn key_value(key: &str, value: &str) {
    println!("  {} : {}", key, value);
}

/// Affiche une paire clé-valeur avec un formatage numérique.
pub fn key_value_numeric(key: &str, value: u64) {
    println!("  {} : {}", key, value);
}

/// Affiche une paire clé-valeur avec un formatage décimal.
pub fn key_value_decimal(key: &str, value: f64) {
    println!("  {} : {:.6}", key, value);
}

/// Affiche un séparateur visuel.
pub fn separator() {
    println!("────────────────────────────────────────────────");
}

/// Affiche un espace vide pour la lisibilité.
pub fn blank_line() {
    println!();
}

/// Affiche un message de progression (pour les opérations longues).
pub fn progress(step: &str, total_steps: usize, current_step: usize) {
    let progress = ((current_step as f64 / total_steps as f64) * 100.0) as u32;
    println!(
        "[{:3}%] Étape {}/{} : {}",
        progress, current_step, total_steps, step
    );
}

/// Affiche les détails d'une opération en mode verbose.
pub fn verbose_details(details: &[(&str, &str)], verbose: bool) {
    if verbose {
        blank_line();
        for (key, value) in details {
            key_value(key, value);
        }
    }
}

/// Affiche un résumé d'opération.
pub fn operation_summary(operation: &str, result: &str, details: Option<&str>) {
    blank_line();
    section(format!("Résumé de l'opération : {}", operation).as_str());
    key_value("Résultat", result);
    if let Some(details) = details {
        key_value("Détails", details);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_level_prefixes() {
        assert_eq!(OutputLevel::Success.prefix(), "SUCCÈS");
        assert_eq!(OutputLevel::Error.prefix(), "ERREUR");
        assert_eq!(OutputLevel::Warning.prefix(), "ATTENTION");
        assert_eq!(OutputLevel::Info.prefix(), "INFO");
        assert_eq!(OutputLevel::Debug.prefix(), "DEBUG");
    }

    #[test]
    fn output_level_stderr_usage() {
        assert!(!OutputLevel::Success.uses_stderr());
        assert!(OutputLevel::Error.uses_stderr());
        assert!(OutputLevel::Warning.uses_stderr());
        assert!(!OutputLevel::Info.uses_stderr());
        assert!(!OutputLevel::Debug.uses_stderr());
    }

    #[test]
    fn output_message_creation() {
        let msg = OutputMessage::new(OutputLevel::Success, "Test message");
        assert_eq!(msg.level, OutputLevel::Success);
        assert_eq!(msg.message, "Test message");
        assert!(msg.advice.is_none());
        assert!(msg.cause.is_none());
    }

    #[test]
    fn output_message_with_advice() {
        let msg = OutputMessage::new(OutputLevel::Error, "Erreur").with_advice("Conseil");
        assert_eq!(msg.advice, Some("Conseil".to_string()));
    }

    #[test]
    fn output_message_with_cause() {
        let msg = OutputMessage::new(OutputLevel::Error, "Erreur").with_cause("Cause technique");
        assert_eq!(msg.cause, Some("Cause technique".to_string()));
    }

    #[test]
    fn output_message_display_format() {
        let msg = OutputMessage::new(OutputLevel::Error, "Test")
            .with_cause("Cause")
            .with_advice("Conseil");
        let display = format!("{}", msg);
        assert!(display.contains("[ERREUR] Test"));
        assert!(display.contains("Cause : Cause"));
        assert!(display.contains("Conseil : Conseil"));
    }

    #[test]
    fn output_message_display_without_optional() {
        let msg = OutputMessage::new(OutputLevel::Info, "Simple message");
        let display = format!("{}", msg);
        assert!(display.contains("[INFO] Simple message"));
        assert!(!display.contains("Cause"));
        assert!(!display.contains("Conseil"));
    }
}
