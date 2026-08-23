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

//! Options globales de la CLI PMG.
//!
//! Ce module définit les options disponibles pour toutes les commandes.
//! Ces options sont héritées par chaque sous-commande et permettent
//! de contrôler le comportement général de l'outil.
//!
//! # Options
//!
//! | Option | Description |
//! |--------|-------------|
//! | `-h, --help` | Affiche l'aide (généré automatiquement par Clap) |
//! | `-d, --dry-run` | Mode sec : simule l'action sans effectuer de modifications |
//! | `--debug` | Active les messages de débogage détaillés |
//! | `-v, --verbose` | Affiche des informations supplémentaires |

use anyhow::Result;
use clap::Parser;
use pmg_core::generator_config::GenerationMode;

/// Options globales communes à toutes les commandes.
///
/// Ces options sont spécifiées après le nom de la commande et avant les arguments
/// spécifiques. Elles contrôlent le comportement général de l'outil.
///
/// # Exemple
///
/// ```bash
/// pmg generate --output model.safetensors --dry-run
/// pmg validate --model-path model.safetensors --verbose
/// pmg compare --original model1.safetensors --compared model2.safetensors --debug
/// ```
#[derive(Debug, Clone, Parser)]
pub struct GlobalOptions {
    /// Mode sec : simule l'action sans effectuer de modifications.
    ///
    /// Lorsque cette option est activée, la commande affiche ce qu'elle
    /// ferait sans réellement exécuter l'opération. Utile pour vérifier
    /// les paramètres avant une exécution réelle.
    #[clap(short = 'd', long = "dry-run", global = true)]
    pub dry_run: bool,

    /// Active les messages de débogage détaillés.
    ///
    /// Affiche des informations techniques supplémentaires utiles pour
    /// le développement et le diagnostic de problèmes.
    #[clap(long = "debug", global = true)]
    pub debug: bool,

    /// Affiche des informations supplémentaires.
    ///
    /// Augmente le niveau de détail des sorties. Peut être combiné avec
    /// d'autres options pour un affichage encore plus verbeux.
    #[clap(short = 'v', long = "verbose", global = true)]
    pub verbose: bool,
}

impl Default for GlobalOptions {
    /// Crée des options globales par défaut (toutes désactivées).
    fn default() -> Self {
        Self {
            dry_run: false,
            debug: false,
            verbose: false,
        }
    }
}

impl GlobalOptions {
    /// Vérifie si un mode de débogage est actif.
    pub fn is_debug_active(&self) -> bool {
        self.debug
    }

    /// Vérifie si le mode verbeux est actif.
    pub fn is_verbose_active(&self) -> bool {
        self.verbose
    }

    /// Vérifie si le mode sec est actif.
    pub fn is_dry_run_active(&self) -> bool {
        self.dry_run
    }

    /// Affiche un résumé des options actives (utile pour le débogage).
    pub fn display_active_options(&self) {
        if self.debug {
            println!("[DEBUG] Options actives :");
            println!("  - dry-run: {}", self.dry_run);
            println!("  - debug: {}", self.debug);
            println!("  - verbose: {}", self.verbose);
        }
    }
}

/// Parse une taille (ex: "1G", "500M", "2T") en octets.
///
/// # Paramètres
/// * `s` - Chaîne de caractères représentant la taille.
///
/// # Retourne
/// La taille en octets ou une erreur si la chaîne est invalide.
///
/// # Exemples
/// ```
/// use pmg_cli::options::parse_size;
///
/// assert_eq!(parse_size("1G").unwrap(), 1_000_000_000);
/// assert_eq!(parse_size("500M").unwrap(), 500_000_000);
/// assert_eq!(parse_size("2T").unwrap(), 2_000_000_000_000);
/// assert_eq!(parse_size("1024").unwrap(), 1024);
/// ```
pub fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim().to_lowercase();
    let (num_str, multiplier) = if let Some(pos) = s.find('t') {
        (&s[..pos], 1_000_000_000_000u64)
    } else if let Some(pos) = s.find('g') {
        (&s[..pos], 1_000_000_000u64)
    } else if let Some(pos) = s.find('m') {
        (&s[..pos], 1_000_000u64)
    } else if let Some(pos) = s.find('k') {
        (&s[..pos], 1_000u64)
    } else {
        (s.as_str(), 1u64)
    };

    let num: f64 = num_str
        .parse()
        .map_err(|_| anyhow::anyhow!("Taille invalide: {}", s))?;

    let result = (num * multiplier as f64) as u64;

    // Validation : la taille ne peut pas être nulle
    if result == 0 {
        return Err(anyhow::anyhow!(
            "La taille ne peut pas être nulle. Spécifiez une taille > 0 (ex: 100M, 1G)"
        ));
    }

    Ok(result)
}

/// Formate une taille en octets en chaîne lisible.
///
/// # Paramètres
/// * `bytes` - Taille en octets.
///
/// # Retourne
/// Chaîne de caractères formatée (ex: "1.50 GB").
///
/// # Exemples
/// ```
/// use pmg_cli::options::format_size;
///
/// assert_eq!(format_size(1_000_000_000), "1.00 GB");
/// assert_eq!(format_size(500_000_000), "500.00 MB");
/// assert_eq!(format_size(2_000_000_000_000), "2.00 TB");
/// ```
pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.2} TB", bytes as f64 / 1_000_000_000_000.0)
    } else if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Parse un mode de génération.
///
/// # Paramètres
/// * `s` - Chaîne de caractères représentant le mode.
///
/// # Retourne
/// Le mode de génération correspondant ou une erreur si le mode n'est pas supporté.
///
/// # Exemples
/// ```
/// use pmg_cli::options::parse_mode;
/// use pmg_core::generator_config::GenerationMode;
///
/// assert_eq!(parse_mode("safe").unwrap(), GenerationMode::Safe);
/// assert_eq!(parse_mode("realistic").unwrap(), GenerationMode::Realistic);
/// assert_eq!(parse_mode("compression").unwrap(), GenerationMode::Compression);
/// assert_eq!(parse_mode("stress").unwrap(), GenerationMode::Stress);
/// ```
pub fn parse_mode(s: &str) -> Result<GenerationMode> {
    match s.to_lowercase().as_str() {
        "safe" => Ok(GenerationMode::Safe),
        "realistic" => Ok(GenerationMode::Realistic),
        "compression" => Ok(GenerationMode::Compression),
        "stress" => Ok(GenerationMode::Stress),
        _ => Err(anyhow::anyhow!("Mode non supporté: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_are_all_false() {
        let opts = GlobalOptions::default();
        assert!(!opts.dry_run);
        assert!(!opts.debug);
        assert!(!opts.verbose);
    }

    #[test]
    fn accessor_methods_work() {
        let mut opts = GlobalOptions::default();
        assert!(!opts.is_dry_run_active());
        assert!(!opts.is_debug_active());
        assert!(!opts.is_verbose_active());

        opts.dry_run = true;
        assert!(opts.is_dry_run_active());

        opts.debug = true;
        assert!(opts.is_debug_active());

        opts.verbose = true;
        assert!(opts.is_verbose_active());
    }

    #[test]
    fn display_does_not_panic() {
        let opts = GlobalOptions::default();
        opts.display_active_options();
    }
}
