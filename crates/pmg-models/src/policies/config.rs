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

//! Sous-module contenant les structures de configuration des politiques.

use std::collections::BTreeMap;

use pmg_core::DType;
use pmg_core::TensorRole;
use serde::{Deserialize, Serialize};

use super::strategies::{
    CompressionStrategy, CorrelationStrategy, LowRankStrategy, OutlierStrategy, SeedStrategy,
};

/// Politique de génération globale pour un modèle.
///
/// Définit les paramètres de base qui s'appliquent à tous les tenseurs
/// lors de la génération, sauf overrides spécifiques.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPolicy {
    /// Nombre d'éléments par chunk de génération (défaut : 1_048_576).
    pub chunk_elements: usize,
    /// Stratégie de dérivation de la seed.
    pub seed_strategy: SeedStrategy,
    /// Si `true`, la génération est strictement déterministe (même seed → même résultat).
    pub deterministic: bool,
    /// Version du générateur utilisée (pour traçabilité).
    pub generator_version: String,
}

impl GenerationPolicy {
    /// Politique par défaut (chunk 1M, seed globale, déterministe).
    pub fn default_policy() -> Self {
        Self {
            chunk_elements: 1_048_576,
            seed_strategy: SeedStrategy::Global,
            deterministic: true,
            generator_version: "1.0.0".to_string(),
        }
    }

    /// Valide les invariants de la politique.
    ///
    /// # Erreurs
    ///
    /// Retourne `Err` si `chunk_elements` est nul.
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_elements == 0 {
            return Err("chunk_elements doit être supérieur à 0".to_string());
        }
        Ok(())
    }
}

/// Politique de types de données pour un modèle.
///
/// Définit le dtype par défaut et les overrides par rôle fonctionnel.
/// Permet d'adapter la précision numérique aux caractéristiques observées
/// de chaque modèle (ex: FP8 pour les experts DeepSeek, BF16 pour GLM).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtypePolicy {
    /// Type de données par défaut pour tous les tenseurs.
    pub default: DType,
    /// Overrides par rôle fonctionnel (ex: `AttentionQuery → F16`).
    pub overrides: BTreeMap<TensorRole, DType>,
}

impl DtypePolicy {
    /// Crée une politique avec un dtype uniforme.
    pub fn uniform(dtype: DType) -> Self {
        Self {
            default: dtype,
            overrides: BTreeMap::new(),
        }
    }

    /// Retourne le dtype effectif pour un rôle donné.
    pub fn effective_dtype(&self, role: TensorRole) -> DType {
        self.overrides.get(&role).copied().unwrap_or(self.default)
    }

    /// Valide les invariants de la politique.
    pub fn validate(&self) -> Result<(), String> {
        // Pas d'invariant structurel fort, mais on vérifie que les overrides
        // ne sont pas vides si fournis (optionnel).
        Ok(())
    }
}

/// Politique de génération par couche.
///
/// Module les paramètres de structure et d'outliers en fonction de l'indice
/// de couche (fonction θ_l = f(l) de la spécification).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerPolicyGlm {
    /// Force de structure de la couche dans [0, 1] (0 = i.i.d., 1 = totalement structuré).
    pub structure_strength: f64,
    /// Intensité des outliers de la couche (multiplicateur d'amplitude).
    pub outlier_intensity: f64,
    /// Densité cible des outliers dans [0, 1].
    pub outlier_density: f64,
    /// Décalage de seed propre à la couche (pour différencier les couches).
    pub layer_seed_shift: u64,
}

impl LayerPolicyGlm {
    /// Politique par défaut pour une couche donnée.
    pub fn default_for(layer_index: u64) -> Self {
        Self {
            structure_strength: 0.1,
            outlier_intensity: 1.0,
            outlier_density: 0.001,
            layer_seed_shift: layer_index.saturating_add(1),
        }
    }

    /// Valide les invariants de la politique.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.structure_strength) {
            return Err("structure_strength doit être dans [0, 1]".to_string());
        }
        if self.outlier_intensity < 0.0 {
            return Err("outlier_intensity doit être ≥ 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.outlier_density) {
            return Err("outlier_density doit être dans [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Politique de configuration des outliers (super-poids).
///
/// Contrôle la fréquence, l'amplitude et la stratégie d'injection
/// des outliers dans les tenseurs générés.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlierPolicy {
    /// Fréquence des outliers dans [0, 1] (probabilité qu'un élément soit un outlier).
    pub frequency: f64,
    /// Échelle multiplicative des outliers (≥ 1 pour amplification).
    pub scale: f64,
    /// Stratégie d'injection des outliers.
    pub strategy: OutlierStrategy,
    /// Degrés de liberté de la Student-t (si strategy = HeavyTail).
    pub heavy_tail_df: f64,
}

impl OutlierPolicy {
    /// Politique par défaut (aucun outlier).
    pub fn none() -> Self {
        Self {
            frequency: 0.0,
            scale: 1.0,
            strategy: OutlierStrategy::Multiplicative,
            heavy_tail_df: 5.0,
        }
    }

    /// Valide les invariants de la politique.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.frequency) {
            return Err("frequency doit être dans [0, 1]".to_string());
        }
        if self.scale < 1.0 {
            return Err("scale doit être ≥ 1".to_string());
        }
        if self.strategy == OutlierStrategy::HeavyTail && self.heavy_tail_df <= 0.0 {
            return Err("heavy_tail_df doit être > 0 pour la stratégie HeavyTail".to_string());
        }
        Ok(())
    }
}

/// Politique de corrélation entre colonnes.
///
/// Configure la force et la stratégie de corrélation appliquée
/// aux tenseurs pour simuler les dépendances inter-colonnes observées.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrelationPolicy {
    /// Force de corrélation dans [0, 1) (0 = indépendant, 1 = parfaitement corrélé).
    pub strength: f64,
    /// Stratégie de corrélation utilisée.
    pub strategy: CorrelationStrategy,
}

impl CorrelationPolicy {
    /// Politique par défaut (aucune corrélation).
    pub fn none() -> Self {
        Self {
            strength: 0.0,
            strategy: CorrelationStrategy::Pearson,
        }
    }

    /// Valide les invariants de la politique.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..1.0).contains(&self.strength) {
            return Err("strength doit être dans [0, 1)".to_string());
        }
        Ok(())
    }
}

/// Politique de décomposition bas-rang.
///
/// Configure l'application de composantes bas-rang pour simuler
/// les structures de faible rang observées dans certains tenseurs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LowRankPolicy {
    /// Seuil de rang en dessous duquel la décomposition est appliquée.
    pub rank_threshold: u64,
    /// Stratégie de décomposition utilisée.
    pub strategy: LowRankStrategy,
    /// Amplitude α de la composante bas-rang (W' = W + α·UVᵀ).
    pub alpha: f64,
    /// Probabilité d'appliquer la décomposition bas-rang.
    pub probability: f64,
}

impl LowRankPolicy {
    /// Politique par défaut (aucune décomposition bas-rang).
    pub fn none() -> Self {
        Self {
            rank_threshold: 1,
            strategy: LowRankStrategy::Svd,
            alpha: 0.0,
            probability: 0.0,
        }
    }

    /// Valide les invariants de la politique.
    pub fn validate(&self) -> Result<(), String> {
        if self.rank_threshold == 0 {
            return Err("rank_threshold doit être > 0".to_string());
        }
        if self.alpha < 0.0 {
            return Err("alpha doit être ≥ 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.probability) {
            return Err("probability doit être dans [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Politique de sérialisation.
///
/// Configure la taille des shards et la stratégie de compression
/// pour l'écriture des fichiers Safetensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializationPolicy {
    /// Taille maximale d'un shard en octets (défaut : 10 Go).
    pub shard_size: u64,
    /// Stratégie de compression appliquée.
    pub compression: CompressionStrategy,
    /// Niveau de compression (0-9, utilisé pour zstd/gzip).
    pub compression_level: u8,
}

impl SerializationPolicy {
    /// Politique par défaut (shard 10 Go, aucune compression).
    pub fn default_policy() -> Self {
        Self {
            shard_size: 10 * 1024 * 1024 * 1024, // 10 Go
            compression: CompressionStrategy::None,
            compression_level: 0,
        }
    }

    /// Valide les invariants de la politique.
    pub fn validate(&self) -> Result<(), String> {
        if self.shard_size == 0 {
            return Err("shard_size doit être > 0".to_string());
        }
        if self.compression_level > 9 {
            return Err("compression_level doit être dans [0, 9]".to_string());
        }
        Ok(())
    }
}

/// Règle de mapping d'un pattern de nom de tenseur vers son rôle et ses politiques.
///
/// Permet d'associer des politiques spécifiques à des tenseurs
/// identifiés par un motif (ex: `model.layers.{layer}.mlp.experts.{expert}.*`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorRule {
    /// Motif du nom de tenseur (supporte les placeholders `{layer}`, `{expert}`).
    pub pattern: String,
    /// Rôle fonctionnel des tenseurs correspondant au motif.
    pub role: TensorRole,
    /// Override du dtype (optionnel).
    pub dtype_override: Option<DType>,
    /// Override de la politique d'outliers (optionnel).
    pub outlier_override: Option<OutlierPolicy>,
    /// Override de la politique de corrélation (optionnel).
    pub correlation_override: Option<CorrelationPolicy>,
    /// Override de la politique bas-rang (optionnel).
    pub low_rank_override: Option<LowRankPolicy>,
}

impl TensorRule {
    /// Crée une règle simple sans overrides.
    pub fn simple(pattern: &str, role: TensorRole) -> Self {
        Self {
            pattern: pattern.to_string(),
            role,
            dtype_override: None,
            outlier_override: None,
            correlation_override: None,
            low_rank_override: None,
        }
    }

    /// Vérifie si un nom de tenseur correspond au motif.
    ///
    /// Pour l'instant, vérifie si le motif est une sous-chaîne du nom.
    /// Une implémentation future pourrait supporter les placeholders `{layer}`.
    pub fn matches(&self, tensor_name: &str) -> bool {
        // Si le pattern ne contient aucun placeholder, vérifier la sous-chaîne
        if !self.pattern.contains('{') {
            return tensor_name.contains(&self.pattern);
        }

        // Séparer le pattern et le nom du tenseur par '.'
        let pattern_segments: Vec<&str> = self.pattern.split('.').collect();
        let tensor_segments: Vec<&str> = tensor_name.split('.').collect();

        // Fonction récursive pour vérifier la correspondance segmentée
        fn matches_segments(pattern: &[&str], tensor: &[&str]) -> bool {
            if pattern.is_empty() {
                return tensor.is_empty();
            }
            let pat = pattern[0];
            if pat.starts_with('{') && pat.ends_with('}') {
                // Placeholder : peut correspondre à un ou plusieurs segments
                for i in 1..=tensor.len() {
                    if matches_segments(&pattern[1..], &tensor[i..]) {
                        return true;
                    }
                }
                false
            } else {
                // Segment literal : doit correspondre exactement
                if tensor.is_empty() {
                    return false;
                }
                if pat != tensor[0] {
                    return false;
                }
                matches_segments(&pattern[1..], &tensor[1..])
            }
        }

        matches_segments(&pattern_segments, &tensor_segments)
    }

    /// Valide les invariants de la règle.
    pub fn validate(&self) -> Result<(), String> {
        if self.pattern.is_empty() {
            return Err("Le pattern ne peut pas être vide".to_string());
        }
        Ok(())
    }
}

/// Politiques globales pour un modèle.
///
/// Regroupe toutes les politiques de génération, de types de données,
/// de corrélation, bas-rang, sérialisation et les règles de mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPolicies {
    /// Politique de génération globale.
    pub generation: GenerationPolicy,
    /// Politique de types de données.
    pub dtype: DtypePolicy,
    /// Politique de corrélation.
    pub correlation: CorrelationPolicy,
    /// Politique bas-rang.
    pub low_rank: LowRankPolicy,
    /// Politique de sérialisation.
    pub serialization: SerializationPolicy,
    /// Règles de mapping pour les tenseurs spécifiques.
    pub tensor_rules: Vec<TensorRule>,
}

impl ModelPolicies {
    /// Crée des politiques par défaut pour un modèle.
    pub fn default_for_model(dtype: DType) -> Self {
        Self {
            generation: GenerationPolicy::default_policy(),
            dtype: DtypePolicy::uniform(dtype),
            correlation: CorrelationPolicy::none(),
            low_rank: LowRankPolicy::none(),
            serialization: SerializationPolicy::default_policy(),
            tensor_rules: Vec::new(),
        }
    }

    /// Valide toutes les politiques.
    pub fn validate(&self) -> Result<(), String> {
        self.generation.validate()?;
        self.dtype.validate()?;
        self.correlation.validate()?;
        self.low_rank.validate()?;
        self.serialization.validate()?;
        for rule in &self.tensor_rules {
            rule.validate()?;
        }
        Ok(())
    }
}
