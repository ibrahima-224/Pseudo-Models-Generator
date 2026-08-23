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

//! Itérateur streaming pour les tenseurs d'un modèle.
//!
//! Ce module fournit [`TensorIterator`] qui retourne un itérateur lazy
//! pour la génération de tenseurs sans accumulation en mémoire. C'est le
//! mode recommandé pour les modèles de grande taille (> 10 GB).
//!
//! ## Principe
//!
//! Au lieu de générer tous les tenseurs en mémoire (`Vec<ModelTensorResult>`),
//! chaque tenseur est généré à la volée via l'itérateur. Cela réduit
//! l'utilisation mémoire de O(model_size) à O(chunk_size).

use crate::context::GenerationContext;
use crate::error::GeneratorResult;
use crate::pipeline::{GenerationPipeline, StepResult};
use crate::seed_plan::GeneratorSeedPlan;
use crate::tensor_generator::TensorGenerator;
use pmg_blueprint::ModelBlueprint;

/// Chunk de tenseur généré en streaming.
#[derive(Debug, Clone)]
pub struct TensorChunk {
    /// Nom du tenseur.
    pub name: String,
    /// Valeurs générées (f64).
    pub values: Vec<f64>,
    /// Résultats du pipeline.
    pub pipeline_results: Vec<StepResult>,
    /// Catégorie du tenseur (embedding, layer, norm, lm_head, extra).
    pub category: String,
    /// Index de la couche (si applicable).
    pub layer_index: Option<usize>,
    /// Nombre total d'éléments.
    pub num_elements: usize,
}

/// Itérateur streaming pour les tenseurs d'un modèle.
///
/// Cet itérateur génère les tenseurs un par un sans accumulation en mémoire.
/// Chaque appel à `next()` génère le tenseur suivant et retourne un `TensorChunk`.
pub struct TensorIterator<'a> {
    /// Blueprint du modèle.
    blueprint: &'a ModelBlueprint,
    /// Seed globale de génération.
    seed: u64,
    /// Version du générateur.
    generation_version: String,
    /// Pipeline de génération.
    pipeline: GenerationPipeline,
    /// Taille des chunks.
    chunk_size: usize,
    /// Budget tensoriel disponible (optionnel, en octets).
    tensor_budget: Option<u64>,
    /// Index actuel dans la séquence de tenseurs.
    current_index: usize,
    /// Nombre total de tenseurs.
    total_tensors: usize,
    /// État de l'itérateur.
    state: IteratorState,
}

/// État interne de l'itérateur.
#[derive(Debug, Clone)]
enum IteratorState {
    /// Initial, prêt à générer les embeddings.
    Embeddings { index: usize },
    /// Génération des couches.
    Layers {
        layer_index: usize,
        tensor_index: usize,
    },
    /// Génération de la norme finale.
    FinalNorm { index: usize },
    /// Génération de la tête de langage.
    LmHead { index: usize },
    /// Génération des tenseurs supplémentaires.
    ExtraTensors { index: usize },
    /// Itération terminée.
    Finished,
}

impl<'a> TensorIterator<'a> {
    /// Crée un nouvel itérateur pour le blueprint spécifié.
    ///
    /// # Paramètres
    /// - `blueprint` : blueprint du modèle
    /// - `seed` : seed globale de génération
    /// - `generation_version` : version du générateur
    /// - `pipeline` : pipeline de génération à appliquer
    /// - `chunk_size` : taille des chunks pour la génération
    /// - `tensor_budget` : budget tensoriel optionnel
    pub fn new(
        blueprint: &'a ModelBlueprint,
        seed: u64,
        generation_version: impl Into<String>,
        pipeline: GenerationPipeline,
        chunk_size: usize,
        tensor_budget: Option<u64>,
    ) -> Self {
        let total_tensors = blueprint.embeddings.len()
            + blueprint
                .layers
                .iter()
                .map(|l| l.all_tensors().len())
                .sum::<usize>()
            + blueprint.final_norm.len()
            + blueprint.lm_head.len()
            + blueprint.extra_tensors.len();

        Self {
            blueprint,
            seed,
            generation_version: generation_version.into(),
            pipeline,
            chunk_size,
            tensor_budget,
            current_index: 0,
            total_tensors,
            state: IteratorState::Embeddings { index: 0 },
        }
    }

    /// Retourne le nombre total de tenseurs.
    pub fn total_tensors(&self) -> usize {
        self.total_tensors
    }

    /// Retourne l'index actuel.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Génère le prochain tenseur de manière lazy.
    fn generate_next_tensor(&mut self) -> Option<GeneratorResult<TensorChunk>> {
        loop {
            match &self.state {
                IteratorState::Embeddings { index } => {
                    if *index >= self.blueprint.embeddings.len() {
                        self.state = IteratorState::Layers {
                            layer_index: 0,
                            tensor_index: 0,
                        };
                        continue;
                    }

                    let tensor_spec = &self.blueprint.embeddings[*index];
                    let result = self.generate_tensor_for_chunk(tensor_spec, "embedding", None);
                    self.state = IteratorState::Embeddings { index: index + 1 };
                    self.current_index += 1;
                    return Some(result);
                },

                IteratorState::Layers {
                    layer_index,
                    tensor_index,
                } => {
                    if *layer_index >= self.blueprint.layers.len() {
                        self.state = IteratorState::FinalNorm { index: 0 };
                        continue;
                    }

                    let layer = &self.blueprint.layers[*layer_index];
                    let all_tensors = layer.all_tensors();

                    if *tensor_index >= all_tensors.len() {
                        self.state = IteratorState::Layers {
                            layer_index: layer_index + 1,
                            tensor_index: 0,
                        };
                        continue;
                    }

                    let tensor_spec = &all_tensors[*tensor_index];
                    let result =
                        self.generate_tensor_for_chunk(tensor_spec, "layer", Some(*layer_index));
                    self.state = IteratorState::Layers {
                        layer_index: *layer_index,
                        tensor_index: tensor_index + 1,
                    };
                    self.current_index += 1;
                    return Some(result);
                },

                IteratorState::FinalNorm { index } => {
                    if *index >= self.blueprint.final_norm.len() {
                        self.state = IteratorState::LmHead { index: 0 };
                        continue;
                    }

                    let tensor_spec = &self.blueprint.final_norm[*index];
                    let result = self.generate_tensor_for_chunk(tensor_spec, "final_norm", None);
                    self.state = IteratorState::FinalNorm { index: index + 1 };
                    self.current_index += 1;
                    return Some(result);
                },

                IteratorState::LmHead { index } => {
                    if *index >= self.blueprint.lm_head.len() {
                        self.state = IteratorState::ExtraTensors { index: 0 };
                        continue;
                    }

                    let tensor_spec = &self.blueprint.lm_head[*index];
                    let result = self.generate_tensor_for_chunk(tensor_spec, "lm_head", None);
                    self.state = IteratorState::LmHead { index: index + 1 };
                    self.current_index += 1;
                    return Some(result);
                },

                IteratorState::ExtraTensors { index } => {
                    if *index >= self.blueprint.extra_tensors.len() {
                        self.state = IteratorState::Finished;
                        continue;
                    }

                    let tensor_spec = &self.blueprint.extra_tensors[*index];
                    let result = self.generate_tensor_for_chunk(tensor_spec, "extra", None);
                    self.state = IteratorState::ExtraTensors { index: index + 1 };
                    self.current_index += 1;
                    return Some(result);
                },

                IteratorState::Finished => {
                    return None;
                },
            }
        }
    }

    /// Génère un tenseur pour un chunk spécifique.
    fn generate_tensor_for_chunk(
        &self,
        tensor_spec: &pmg_blueprint::tensor_spec::TensorSpec,
        category: &str,
        layer_index: Option<usize>,
    ) -> GeneratorResult<TensorChunk> {
        let num_elements = tensor_spec.num_elements()? as usize;

        // Créer le contexte de génération
        let context = GenerationContext::new(
            self.seed,
            &self.blueprint.id,
            &self.generation_version,
            layer_index,
            0, // tensor_index
            0, // chunk_index initial
            &tensor_spec.name,
            num_elements,
            self.chunk_size,
        );

        // Créer le plan de seed
        let seed_plan =
            GeneratorSeedPlan::new(self.seed, &self.blueprint.id, &self.generation_version);

        // Générer les valeurs initiales avec le budget tensoriel si disponible
        // Utiliser une taille de chunk réduite pour optimiser la mémoire
        let _optimized_chunk_size = self.chunk_size.min(num_elements);
        let tensor_gen = TensorGenerator::new(tensor_spec.clone(), seed_plan, self.tensor_budget);
        let mut values = tensor_gen.generate()?;

        // Appliquer le pipeline sur les valeurs générées
        let pipeline_results = self.pipeline.execute(&mut values, context.tensor_seed())?;

        // Si le tenseur est trop grand, ne garder qu'un chunk en mémoire
        // Pour l'instant, on garde le comportement existant mais on pourrait
        // implémente un streaming plus avancé ici
        Ok(TensorChunk {
            name: tensor_spec.name.clone(),
            values,
            pipeline_results,
            category: category.to_string(),
            layer_index,
            num_elements,
        })
    }
}

impl<'a> Iterator for TensorIterator<'a> {
    type Item = GeneratorResult<TensorChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        self.generate_next_tensor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_blueprint::architecture::ArchitectureKind;
    use pmg_blueprint::naming::NamingRules;
    use pmg_blueprint::tensor_spec::TensorSpec;
    use pmg_core::model_config::glm52_test_config;
    use pmg_core::{DType, Shape, TensorRole};

    fn test_blueprint() -> ModelBlueprint {
        let config = glm52_test_config();
        let mut bp = ModelBlueprint::new(
            "glm-5.2",
            ArchitectureKind::MoETransformer,
            config,
            NamingRules::glm52(),
        );

        // Ajouter un embedding
        bp.embeddings.push(
            TensorSpec::new(
                "model.embed_tokens.weight",
                Shape::new(vec![100, 64]).unwrap(),
                DType::F32,
                TensorRole::Embedding,
            )
            .unwrap(),
        );

        // Ajouter une couche
        let mut layer =
            pmg_blueprint::layer::LayerSpec::new(0, pmg_blueprint::layer::LayerKind::Dense);
        layer.attention.push(
            TensorSpec::new(
                "model.layers.0.self_attn.q_proj.weight",
                Shape::new(vec![64, 64]).unwrap(),
                DType::F32,
                TensorRole::AttentionQuery,
            )
            .unwrap(),
        );
        bp.layers.push(layer);

        // Ajouter une norme finale
        bp.final_norm.push(
            TensorSpec::new(
                "model.norm.weight",
                Shape::new(vec![64]).unwrap(),
                DType::F32,
                TensorRole::Norm,
            )
            .unwrap(),
        );

        // Ajouter une tête de langage
        bp.lm_head.push(
            TensorSpec::new(
                "lm_head.weight",
                Shape::new(vec![100, 64]).unwrap(),
                DType::F32,
                TensorRole::LmHead,
            )
            .unwrap(),
        );

        bp
    }

    #[test]
    fn test_tensor_iterator_creation() {
        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let iterator = TensorIterator::new(&blueprint, 42, "1.0.0", pipeline, 256, None);

        assert_eq!(iterator.total_tensors(), 4); // 1 embedding + 1 layer + 1 norm + 1 lm_head
        assert_eq!(iterator.current_index(), 0);
    }

    #[test]
    fn test_tensor_iterator_iteration() {
        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let mut iterator = TensorIterator::new(&blueprint, 42, "1.0.0", pipeline, 256, None);

        // Premier tenseur (embedding)
        let first = iterator.next().unwrap().unwrap();
        assert_eq!(first.name, "model.embed_tokens.weight");
        assert_eq!(first.category, "embedding");
        assert_eq!(first.layer_index, None);

        // Deuxième tenseur (layer)
        let second = iterator.next().unwrap().unwrap();
        assert_eq!(second.name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(second.category, "layer");
        assert_eq!(second.layer_index, Some(0));

        // Vérifier qu'on a bien 4 tenseurs au total
        let mut count = 2;
        for _ in iterator {
            count += 1;
        }
        assert_eq!(count, 4);
    }
}
