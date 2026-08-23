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

//! Génération par streaming pour la gestion mémoire optimisée.
//!
//! Ce module fournit le flux de génération :
//! `Generate chunk → Transform chunk → Encode chunk → Write chunk → Discard chunk`
//!
//! Conformité : ADR-002, étape 4 - Déplacement de streaming_generation.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_generator::streaming::{StreamingGenerator, Chunk};
//! use pmg_generator::generator_config::GeneratorConfig;
//! use pmg_core::generation_plan::GenerationPlan;
//! use pmg_core::shape::Shape;
//! use pmg_core::dtype::DType;
//! use pmg_core::rng_trait::DeterministicRng;
//!
//! // RNG de démonstration (en production, utilise ChaCha12 via pmg_math)
//! #[derive(Debug)]
//! struct MockRng(u64);
//! impl DeterministicRng for MockRng {
//!     fn next_u64(&mut self) -> u64 {
//!         self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//!         self.0
//!     }
//!     fn next_f64(&mut self) -> f64 { (self.next_u64() as f64) / (u64::MAX as f64) }
//! }
//!
//! let config = GeneratorConfig::default();
//! let shape = Shape::new(vec![100, 100]).unwrap();
//! let plan = GenerationPlan::new("test.tensor", shape, DType::F32, 42).unwrap();
//! let rng = Box::new(MockRng(42));
//!
//! let mut generator = StreamingGenerator::new(config, plan, rng);
//! // Génère par chunks de 1024 éléments
//! while let Some(chunk) = generator.next_chunk().unwrap() {
//!     // Traite le chunk (encodage, écriture, etc.)
//!     assert!(chunk.elements.len() <= 1024);
//! }
//! ```

use pmg_core::error::CoreResult;
use pmg_core::generation_plan::GenerationPlan;
use pmg_core::rng_trait::DeterministicRng;

use crate::generator_config::GeneratorConfig;

/// Taille par défaut des chunks (éléments).
pub const DEFAULT_CHUNK_SIZE: usize = 1024;

/// État du streaming pour un tenseur.
#[derive(Debug)]
pub struct StreamingGenerator {
    /// Configuration de génération.
    config: GeneratorConfig,
    /// Plan du tenseur en cours.
    plan: GenerationPlan,
    /// Taille des chunks.
    chunk_size: usize,
    /// Index du prochain chunk.
    next_chunk_index: usize,
    /// Nombre total d'éléments.
    total_elements: usize,
    /// Générateur aléatoire déterministe.
    rng: Box<dyn DeterministicRng>,
}

impl StreamingGenerator {
    /// Crée un nouveau générateur streaming avec un RNG déterministe.
    ///
    /// # Paramètres
    /// - `config` : configuration de génération.
    /// - `plan` : plan du tenseur.
    /// - `rng` : générateur aléatoire déterministe à utiliser.
    pub fn new(
        config: GeneratorConfig,
        plan: GenerationPlan,
        rng: Box<dyn DeterministicRng>,
    ) -> Self {
        let total_elements = plan.num_elements().unwrap_or(0) as usize;
        let chunk_size = plan.chunk_elements.unwrap_or(DEFAULT_CHUNK_SIZE as u64) as usize;

        Self {
            config,
            plan,
            chunk_size,
            next_chunk_index: 0,
            total_elements,
            rng,
        }
    }

    /// Crée un générateur avec une taille de chunk spécifique.
    pub fn with_chunk_size(
        config: GeneratorConfig,
        plan: GenerationPlan,
        chunk_size: usize,
        rng: Box<dyn DeterministicRng>,
    ) -> Self {
        let total_elements = plan.num_elements().unwrap_or(0) as usize;
        Self {
            config,
            plan,
            chunk_size,
            next_chunk_index: 0,
            total_elements,
            rng,
        }
    }

    /// Génère le prochain chunk de valeurs.
    ///
    /// Retourne `None` quand tous les éléments ont été générés.
    pub fn next_chunk(&mut self) -> CoreResult<Option<Chunk>> {
        if self.next_chunk_index * self.chunk_size >= self.total_elements {
            return Ok(None);
        }

        let start = self.next_chunk_index * self.chunk_size;
        let end = std::cmp::min(start + self.chunk_size, self.total_elements);
        let chunk_size = end - start;

        // Génère les valeurs pour ce chunk
        let mut values = Vec::with_capacity(chunk_size);
        for i in 0..chunk_size {
            let _global_index = start + i;
            let value = self.generate_element()?;
            values.push(value);
        }

        let chunk = Chunk {
            index: self.next_chunk_index,
            start,
            end,
            elements: values,
        };

        self.next_chunk_index += 1;
        Ok(Some(chunk))
    }

    /// Génère un élément unique à son index global.
    fn generate_element(&mut self) -> CoreResult<f64> {
        // Utilise le RNG déterministe pour générer la valeur
        // Note: nous utilisons next_f64 qui retourne une valeur dans [0, 1)
        let value = self.rng.next_f64();
        Ok(value)
    }

    /// Retourne la progression de la génération.
    pub fn progress(&self) -> f64 {
        if self.total_elements == 0 {
            return 1.0;
        }
        let generated = self.next_chunk_index * self.chunk_size;
        (generated as f64) / (self.total_elements as f64)
    }

    /// Vérifie si la génération est terminée.
    pub fn is_complete(&self) -> bool {
        self.next_chunk_index * self.chunk_size >= self.total_elements
    }

    /// Retourne le nombre total d'éléments.
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Retourne la taille des chunks.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Retourne la configuration.
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Retourne le plan.
    pub fn plan(&self) -> &GenerationPlan {
        &self.plan
    }
}

/// Chunk de données généré.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Index du chunk (0-based).
    pub index: usize,
    /// Index de début global.
    pub start: usize,
    /// Index de fin global (exclus).
    pub end: usize,
    /// Valeurs du chunk.
    pub elements: Vec<f64>,
}

impl Chunk {
    /// Nombre d'éléments dans le chunk.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Vérifie si le chunk est vide.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Convertit le chunk en tenseur généré (pour compatibilité).
    pub fn to_tensor(&self, tensor_name: &str) -> crate::generator_core::GeneratedTensor {
        crate::generator_core::GeneratedTensor::new(tensor_name, self.elements.clone())
    }
}

// NOTE: La structure ChunkIterator est définie dans chunk.rs
// et utilise un closure-based pattern plus flexible.
// Nous fournissons une fonction utilitaire pour créer un itérateur de chunks
// à partir d'un StreamingGenerator.

/// Statistiques de streaming.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Nombre de chunks générés.
    pub chunks_generated: usize,
    /// Nombre total d'éléments.
    pub total_elements: usize,
    /// Taille moyenne des chunks.
    pub avg_chunk_size: f64,
    /// Mémoire maximale utilisée (estimation).
    pub peak_memory_bytes: usize,
}

impl StreamingStats {
    /// Crée de nouvelles statistiques à partir d'un générateur.
    pub fn from_generator(generator: &StreamingGenerator) -> Self {
        let chunks_generated = generator.next_chunk_index;
        let total_elements = generator.total_elements;
        let avg_chunk_size = if chunks_generated > 0 {
            total_elements as f64 / chunks_generated as f64
        } else {
            0.0
        };

        // Estimation de la mémoire : taille d'un chunk × 8 octets (f64)
        let peak_memory_bytes = generator.chunk_size * 8;

        Self {
            chunks_generated,
            total_elements,
            avg_chunk_size,
            peak_memory_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::dtype::DType;
    use pmg_core::shape::Shape;

    /// RNG de test simple pour vérifier le fonctionnement
    #[derive(Debug)]
    struct MockRng {
        state: u64,
    }

    impl MockRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
    }

    impl DeterministicRng for MockRng {
        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn next_f64(&mut self) -> f64 {
            (self.next_u64() as f64) / (u64::MAX as f64)
        }
    }

    #[test]
    fn streaming_generator_basic() {
        let config = GeneratorConfig::default();
        let shape = Shape::new(vec![100, 100]).unwrap();
        let plan = GenerationPlan::new("test.tensor", shape, DType::F32, 42).unwrap();
        let rng = Box::new(MockRng::new(42));

        let mut generator = StreamingGenerator::new(config, plan, rng);
        let mut total_elements = 0;

        while let Some(chunk) = generator.next_chunk().unwrap() {
            total_elements += chunk.len();
            assert!(chunk.len() <= DEFAULT_CHUNK_SIZE);
        }

        assert_eq!(total_elements, 10_000);
        assert!(generator.is_complete());
    }

    #[test]
    fn streaming_generator_custom_chunk_size() {
        let config = GeneratorConfig::default();
        let shape = Shape::new(vec![10, 10]).unwrap();
        let plan = GenerationPlan::new("test.tensor", shape, DType::F32, 42).unwrap();
        let rng = Box::new(MockRng::new(42));

        let mut generator = StreamingGenerator::with_chunk_size(config, plan, 25, rng);
        let mut total_elements = 0;

        while let Some(chunk) = generator.next_chunk().unwrap() {
            total_elements += chunk.len();
            assert!(chunk.len() <= 25);
        }

        assert_eq!(total_elements, 100);
        assert!(generator.is_complete());
    }

    #[test]
    fn streaming_generator_with_while_let() {
        let config = GeneratorConfig::default();
        let shape = Shape::new(vec![10, 10]).unwrap();
        let plan = GenerationPlan::new("test.tensor", shape, DType::F32, 42).unwrap();
        let rng = Box::new(MockRng::new(42));

        let mut generator = StreamingGenerator::new(config, plan, rng);
        let mut chunks = Vec::new();

        while let Some(chunk) = generator.next_chunk().unwrap() {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 1); // 100 éléments / 1024 taille chunk = 1 chunk
        assert_eq!(chunks[0].len(), 100);
    }

    #[test]
    fn streaming_stats() {
        let config = GeneratorConfig::default();
        let shape = Shape::new(vec![10, 10]).unwrap();
        let plan = GenerationPlan::new("test.tensor", shape, DType::F32, 42).unwrap();
        let rng = Box::new(MockRng::new(42));

        let mut generator = StreamingGenerator::new(config, plan, rng);
        while generator.next_chunk().unwrap().is_some() {}

        let stats = StreamingStats::from_generator(&generator);
        assert_eq!(stats.chunks_generated, 1);
        assert_eq!(stats.total_elements, 100);
        assert!(stats.peak_memory_bytes > 0);
    }
}
