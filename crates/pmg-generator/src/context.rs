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

//! Contexte de génération pour un tenseur.
//!
//! Ce module définit la structure `GenerationContext` qui regroupe toutes les
//! informations contextuelles nécessaires à la génération déterministe d'un tenseur.
//! Le contexte permet de dériver de manière reproductible les seeds et de
//! tracer l'origine de chaque valeur générée.

/// Contexte de génération pour un tenseur donné.
///
/// Contient toutes les informations permettant de dériver de manière déterministe
/// les seeds et de tracer la génération. Le contexte est immuable une fois créé.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationContext {
    /// Seed globale de génération (fournie par l'utilisateur).
    pub seed: u64,
    /// Identifiant du modèle (ex: "glm-5.2").
    pub model_name: String,
    /// Version du générateur (ex: "1.0.0").
    pub generation_version: String,
    /// Index de la couche (0-based, None pour les embeddings/normes finales).
    pub layer_index: Option<usize>,
    /// Index du tenseur dans la couche (0-based).
    pub tensor_index: usize,
    /// Index du chunk (0-based, pour la génération par chunks).
    pub chunk_index: usize,
    /// Nom complet du tenseur (ex: "model.layers.0.self_attn.q_proj.weight").
    pub tensor_name: String,
    /// Nombre total d'éléments du tenseur.
    pub num_elements: usize,
    /// Taille des chunks utilisée pour la génération.
    pub chunk_size: usize,
}

impl GenerationContext {
    /// Crée un nouveau contexte de génération.
    ///
    /// # Paramètres
    /// - `seed` : seed globale de génération
    /// - `model_name` : identifiant du modèle
    /// - `generation_version` : version du générateur
    /// - `layer_index` : index de la couche (None pour les tenseurs hors couche)
    /// - `tensor_index` : index du tenseur dans la couche
    /// - `chunk_index` : index du chunk en cours de génération
    /// - `tensor_name` : nom complet du tenseur
    /// - `num_elements` : nombre total d'éléments du tenseur
    /// - `chunk_size` : taille des chunks
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seed: u64,
        model_name: impl Into<String>,
        generation_version: impl Into<String>,
        layer_index: Option<usize>,
        tensor_index: usize,
        chunk_index: usize,
        tensor_name: impl Into<String>,
        num_elements: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            seed,
            model_name: model_name.into(),
            generation_version: generation_version.into(),
            layer_index,
            tensor_index,
            chunk_index,
            tensor_name: tensor_name.into(),
            num_elements,
            chunk_size,
        }
    }

    /// Déduit la seed spécifique au tenseur à partir du contexte.
    ///
    /// Cette méthode utilise une dérivation déterministe basée sur tous les
    /// champs du contexte pour produire une seed unique pour ce tenseur.
    pub fn tensor_seed(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.model_name.hash(&mut hasher);
        self.generation_version.hash(&mut hasher);
        self.layer_index.hash(&mut hasher);
        self.tensor_index.hash(&mut hasher);
        self.tensor_name.hash(&mut hasher);
        hasher.finish()
    }

    /// Déduit la seed spécifique à un chunk à partir du contexte.
    ///
    /// La seed du chunk est dérivée de la seed du tenseur et de l'index du chunk.
    pub fn chunk_seed(&self, chunk_id: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let tensor_seed = self.tensor_seed();
        let mut hasher = DefaultHasher::new();
        tensor_seed.hash(&mut hasher);
        chunk_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Retourne le nombre total de chunks nécessaires pour ce tenseur.
    pub fn total_chunks(&self) -> usize {
        self.num_elements.div_ceil(self.chunk_size)
    }

    /// Retourne la plage d'éléments pour un chunk donné.
    ///
    /// # Retourne
    /// Un tuple `(start, end)` où `start` est inclus et `end` exclus.
    pub fn chunk_range(&self, chunk_id: usize) -> (usize, usize) {
        let start = chunk_id * self.chunk_size;
        let end = std::cmp::min(start + self.chunk_size, self.num_elements);
        (start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_creation() {
        let ctx = GenerationContext::new(
            42,
            "glm-5.2",
            "1.0.0",
            Some(0),
            0,
            0,
            "model.layers.0.self_attn.q_proj.weight",
            1024,
            256,
        );

        assert_eq!(ctx.seed, 42);
        assert_eq!(ctx.model_name, "glm-5.2");
        assert_eq!(ctx.generation_version, "1.0.0");
        assert_eq!(ctx.layer_index, Some(0));
        assert_eq!(ctx.tensor_index, 0);
        assert_eq!(ctx.chunk_index, 0);
        assert_eq!(ctx.tensor_name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(ctx.num_elements, 1024);
        assert_eq!(ctx.chunk_size, 256);
    }

    #[test]
    fn tensor_seed_deterministic() {
        let ctx1 =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        let ctx2 =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        assert_eq!(ctx1.tensor_seed(), ctx2.tensor_seed());
    }

    #[test]
    fn different_contexts_different_seeds() {
        let ctx1 =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        let ctx2 =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 1, 0, "tensor_a", 100, 32);
        assert_ne!(ctx1.tensor_seed(), ctx2.tensor_seed());
    }

    #[test]
    fn chunk_seed_deterministic() {
        let ctx =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        let seed1 = ctx.chunk_seed(0);
        let seed2 = ctx.chunk_seed(0);
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn different_chunks_different_seeds() {
        let ctx =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        let seed0 = ctx.chunk_seed(0);
        let seed1 = ctx.chunk_seed(1);
        assert_ne!(seed0, seed1);
    }

    #[test]
    fn total_chunks_calculation() {
        let ctx =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        assert_eq!(ctx.total_chunks(), 4); // 100/32 = 3.125 → 4 chunks
    }

    #[test]
    fn chunk_range() {
        let ctx =
            GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
        assert_eq!(ctx.chunk_range(0), (0, 32));
        assert_eq!(ctx.chunk_range(1), (32, 64));
        assert_eq!(ctx.chunk_range(2), (64, 96));
        assert_eq!(ctx.chunk_range(3), (96, 100)); // dernier chunk tronqué
    }
}
