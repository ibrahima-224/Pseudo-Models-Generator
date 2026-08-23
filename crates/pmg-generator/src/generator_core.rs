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

//! Structures de base et méthodes simples du générateur.
//!
//! Ce module contient les types fondamentaux (`GenerationConfig`, `GeneratedTensor`,
//! `ModelGenerator`) ainsi que les méthodes de configuration et d'accès.
//! Les méthodes de génération lourdes sont implémentées dans `generator_impl`.

use half::f16;
use pmg_blueprint::{ModelBlueprint, Plan};
use pmg_core::generation_plan::GenerationPlan;
use pmg_core::rng_trait::DeterministicRng;
use pmg_injector::injection_policy::InjectionPolicy;

use crate::budget::{BudgetPlanner, GenerationMode};
use crate::chunk::DEFAULT_CHUNK_SIZE;
use crate::error::{GeneratorError, GeneratorResult};
use crate::lazy_iterator::LazyBaseDistribution;
use crate::pipeline::GenerationPipeline;

/// Configuration de génération.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Seed globale de génération.
    pub seed: u64,
    /// Identifiant du modèle.
    pub model_id: String,
    /// Version du générateur.
    pub generation_version: String,
    /// Taille des chunks (éléments par chunk).
    pub chunk_size: usize,
    /// Politique d'injection par défaut.
    pub injection_policy: InjectionPolicy,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            model_id: "unknown".to_string(),
            generation_version: "1.0.0".to_string(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            injection_policy: InjectionPolicy::default(),
        }
    }
}

/// Résultat de la génération d'un tenseur.
#[derive(Debug, Clone)]
pub struct GeneratedTensor {
    /// Nom du tenseur.
    pub name: String,
    /// Valeurs générées (f64).
    pub values: Vec<f64>,
    /// Nombre d'éléments.
    pub num_elements: usize,
}

impl GeneratedTensor {
    /// Crée un nouveau tenseur généré.
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        let num_elements = values.len();
        Self {
            name: name.into(),
            values,
            num_elements,
        }
    }

    /// Convertit les valeurs f64 en bytes selon le dtype cible.
    ///
    /// # Paramètres
    /// - `dtype` : type de données de sortie.
    ///
    /// # Retourne
    /// Un vecteur d'octets contenant les valeurs sérialisées.
    ///
    /// # Erreurs
    /// Retourne une erreur si le dtype n'est pas supporté.
    pub fn to_bytes(&self, dtype: pmg_core::DType) -> pmg_core::CoreResult<Vec<u8>> {
        match dtype {
            pmg_core::DType::F32 => {
                let mut bytes = Vec::with_capacity(self.num_elements * 4);
                for &value in &self.values {
                    let f32_value = value as f32;
                    bytes.extend_from_slice(&f32_value.to_le_bytes());
                }
                Ok(bytes)
            },
            pmg_core::DType::F16 => {
                // Conversion correcte f64 → f32 → f16 via la crate `half`
                // La precision de f16 est limitée (≈3 décimales), donc on passe par f32
                // comme étape intermédiaire pour gérer l'arrondi avant la troncation.
                let mut bytes = Vec::with_capacity(self.num_elements * 2);
                for &value in &self.values {
                    let f32_value = value as f32;
                    let f16_value = f16::from_f32(f32_value);
                    bytes.extend_from_slice(&f16_value.to_le_bytes());
                }
                Ok(bytes)
            },
            _ => Err(pmg_core::CoreError::UnsupportedDType(format!(
                "conversion non supportée pour {:?}",
                dtype
            ))),
        }
    }
}

/// Wrapper pour adapter une référence mutable à un `Box<dyn DeterministicRng>`.
///
/// Ce wrapper permet d'utiliser `LazyBaseDistribution` qui attend un `Box<dyn DeterministicRng>`
/// tout en conservant la signature existante de `generate_tensor` qui prend une référence mutable.
struct DeterministicRngWrapper<'a> {
    /// Référence mutable au générateur déterministe sous-jacent.
    rng: &'a mut dyn DeterministicRng,
}

impl<'a> DeterministicRngWrapper<'a> {
    /// Crée un nouveau wrapper autour d'une référence mutable au RNG.
    fn new(rng: &'a mut dyn DeterministicRng) -> Self {
        Self { rng }
    }
}

impl<'a> DeterministicRng for DeterministicRngWrapper<'a> {
    /// Génère un entier non signé 64 bits aléatoire en déléguant au RNG sous-jacent.
    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Génère un flottant 64 bits aléatoire dans [0, 1) en déléguant au RNG sous-jacent.
    fn next_f64(&mut self) -> f64 {
        self.rng.next_f64()
    }
}

impl<'a> std::fmt::Debug for DeterministicRngWrapper<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeterministicRngWrapper").finish()
    }
}

/// Génère les valeurs d'un tenseur à partir de son plan.
///
/// # Paramètres
/// - `plan` : plan de génération du tenseur.
/// - `rng` : générateur aléatoire déterministe à utiliser.
///
/// # Retourne
/// Un vecteur de `f64` contenant les valeurs générées.
///
/// # Erreurs
/// Retourne une erreur si le plan est invalide ou si la génération échoue.
pub fn generate_tensor<'a>(
    plan: &GenerationPlan,
    rng: &'a mut dyn DeterministicRng,
) -> GeneratorResult<Vec<f64>> {
    // Valide le plan
    plan.validate()
        .map_err(|e| GeneratorError::InvalidModelConfig(format!("plan invalide : {e}")))?;

    let num_elements = plan.num_elements().map_err(|e| {
        GeneratorError::InvalidModelConfig(format!("nombre d'éléments invalide : {e}"))
    })? as usize;

    // Étape X₀ : distribution de base avec LazyBaseDistribution pour optimisation mémoire
    let rng_box: Box<dyn DeterministicRng + 'a> = Box::new(DeterministicRngWrapper::new(rng));
    let values: Vec<f64> = LazyBaseDistribution::new(num_elements, rng_box).collect();

    // NOTE: Les étapes X₁ (structure), X₂ (corrélation), X₃ (outliers)
    // seront implémentées dans le pipeline de génération complet.

    Ok(values)
}

/// Génère la distribution de base (X₀) avec un RNG déterministe.
///
/// # Paramètres
/// - `num_elements` : nombre d'éléments à générer.
/// - `rng` : générateur aléatoire déterministe à utiliser.
///
/// # Retourne
/// Un vecteur de `f64` contenant les valeurs générées (uniformes entre 0 et 1).
///
/// # Notes
///
/// Cette fonction utilise le RNG fourni (normalement ChaCha12) pour garantir
/// le déterminisme et la reproductibilité. La distribution est uniforme sur [0, 1).
///
/// # Optimisation Mémoire
/// Cette fonction est maintenue pour la rétrocompatibilité mais alloue un Vec
/// contenant TOUS les éléments. Pour les distributions de grande taille,
/// préférez l'utilisation directe de `LazyBaseDistribution`.
#[deprecated(
    since = "2.0.0",
    note = "Utiliser LazyBaseDistribution pour l'optimisation mémoire"
)]
#[allow(dead_code)]
fn generate_base_distribution(
    num_elements: usize,
    rng: &mut dyn DeterministicRng,
) -> GeneratorResult<Vec<f64>> {
    // Collecte directe pour la rétrocompatibilité
    // Note: Cette fonction alloue un Vec contenant TOUS les éléments.
    // Pour les distributions de grande taille, utilisez LazyBaseDistribution directement.
    let mut values = Vec::with_capacity(num_elements);
    for _ in 0..num_elements {
        values.push(rng.next_f64());
    }
    Ok(values)
}

/// Orchestrateur principal de génération.
pub struct ModelGenerator {
    /// Configuration de génération.
    pub(crate) config: GenerationConfig,
    /// Blueprint du modèle (si fourni).
    pub(crate) blueprint: Option<ModelBlueprint>,
    /// Plan de tenseurs (si disponible).
    pub(crate) plan: Option<Plan>,
    /// Pipeline de génération (Sprint 12).
    pub(crate) pipeline: GenerationPipeline,
    /// Planner de budget D8.
    pub(crate) budget_planner: BudgetPlanner,
    /// Mode de génération actuel.
    pub(crate) generation_mode: GenerationMode,
    /// Budget tensoriel calculé (optionnel, en octets).
    pub(crate) tensor_budget: Option<u64>,
}

impl ModelGenerator {
    /// Crée un nouveau générateur avec une configuration donnée.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::{ModelGenerator, GenerationConfig};
    ///
    /// let config = GenerationConfig {
    ///     seed: 42,
    ///     model_id: "glm-5.2".to_string(),
    ///     ..GenerationConfig::default()
    /// };
    ///
    /// let gen = ModelGenerator::new(config);
    /// assert_eq!(gen.config().seed, 42);
    /// ```
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            config,
            blueprint: None,
            plan: None,
            pipeline: GenerationPipeline::full(),
            budget_planner: BudgetPlanner::default(),
            generation_mode: GenerationMode::FullStructural,
            tensor_budget: None,
        }
    }

    /// Crée un générateur avec une seed et un model_id simples.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ModelGenerator;
    ///
    /// let gen = ModelGenerator::with_seed(123, "glm-5.2");
    /// assert_eq!(gen.config().seed, 123);
    /// assert_eq!(gen.config().model_id, "glm-5.2");
    /// ```
    pub fn with_seed(seed: u64, model_id: impl Into<String>) -> Self {
        Self::new(GenerationConfig {
            seed,
            model_id: model_id.into(),
            ..GenerationConfig::default()
        })
    }

    /// Définit le blueprint du modèle.
    ///
    /// Le blueprint définit la structure du modèle (tenseurs, couches, etc.)
    /// et doit être défini avant d'appeler `plan()`.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ModelGenerator;
    /// use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    /// use pmg_core::model_config::glm52_test_config;
    ///
    /// let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    ///
    /// let mut bp = ModelBlueprint::new(
    ///     "glm-5.2",
    ///     ArchitectureKind::MoETransformer,
    ///     glm52_test_config(),
    ///     NamingRules::glm52(),
    /// );
    ///
    /// gen.set_blueprint(bp);
    /// assert!(gen.blueprint().is_some());
    /// ```
    pub fn set_blueprint(&mut self, blueprint: ModelBlueprint) {
        self.blueprint = Some(blueprint);
    }

    /// Définit le pipeline de génération.
    ///
    /// Le pipeline contrôle les étapes de transformation appliquées aux valeurs
    /// générées. Par défaut, le pipeline complet est utilisé.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::{ModelGenerator, GenerationPipeline, PipelineStep};
    ///
    /// let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    /// let mut pipeline = GenerationPipeline::empty();
    /// pipeline.add_step(PipelineStep::Distribution);
    ///
    /// gen.set_pipeline(pipeline);
    /// assert_eq!(gen.pipeline().step_count(), 1);
    /// ```
    pub fn set_pipeline(&mut self, pipeline: GenerationPipeline) {
        self.pipeline = pipeline;
    }

    /// Retourne une référence au pipeline de génération.
    pub fn pipeline(&self) -> &GenerationPipeline {
        &self.pipeline
    }

    /// Définit le planner de budget D8.
    pub fn set_budget_planner(&mut self, planner: BudgetPlanner) {
        self.budget_planner = planner;
    }

    /// Retourne une référence au planner de budget.
    pub fn budget_planner(&self) -> &BudgetPlanner {
        &self.budget_planner
    }

    /// Définit le mode de génération.
    pub fn set_generation_mode(&mut self, mode: GenerationMode) {
        self.generation_mode = mode;
    }

    /// Définit le budget tensoriel calculé.
    pub fn set_tensor_budget(&mut self, budget: Option<u64>) {
        self.tensor_budget = budget;
    }

    /// Retourne le budget tensoriel calculé.
    pub fn tensor_budget(&self) -> Option<u64> {
        self.tensor_budget
    }

    /// Retourne le mode de génération actuel.
    pub fn generation_mode(&self) -> &GenerationMode {
        &self.generation_mode
    }

    /// Retourne la configuration de génération.
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Retourne le blueprint si défini.
    pub fn blueprint(&self) -> Option<&ModelBlueprint> {
        self.blueprint.as_ref()
    }

    /// Retourne le plan si disponible.
    pub fn plan_ref(&self) -> Option<&Plan> {
        self.plan.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    use pmg_core::model_config::glm52_test_config;

    fn test_blueprint() -> ModelBlueprint {
        let config = glm52_test_config();
        let mut bp = ModelBlueprint::new(
            "glm-5.2",
            ArchitectureKind::MoETransformer,
            config,
            NamingRules::glm52(),
        );
        // Ajouter un tenseur de test
        bp.embeddings.push(
            pmg_blueprint::TensorSpec::new(
                "model.embed_tokens.weight",
                pmg_core::Shape::new(vec![100, 64]).unwrap(),
                pmg_core::DType::F32,
                pmg_core::TensorRole::Embedding,
            )
            .unwrap(),
        );
        bp
    }

    #[test]
    fn generator_creation() {
        let config = GenerationConfig::default();
        let gen = ModelGenerator::new(config);
        assert_eq!(gen.config().seed, 42);
    }

    #[test]
    fn generator_with_seed() {
        let gen = ModelGenerator::with_seed(123, "test-model");
        assert_eq!(gen.config().seed, 123);
        assert_eq!(gen.config().model_id, "test-model");
    }

    #[test]
    fn generator_with_blueprint() {
        let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
        let blueprint = test_blueprint();
        gen.set_blueprint(blueprint);
        assert!(gen.blueprint().is_some());
    }

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
    fn generate_tensor_basic() {
        let shape = pmg_core::Shape::new(vec![10, 10]).unwrap();
        let plan = GenerationPlan::new("test.tensor", shape, pmg_core::DType::F32, 42).unwrap();
        let mut rng = MockRng::new(42);
        let values = generate_tensor(&plan, &mut rng).unwrap();
        assert_eq!(values.len(), 100);
        // Vérifie que les valeurs sont dans [0, 1]
        for &v in &values {
            assert!((0.0..=1.0).contains(&v), "valeur hors bornes : {v}");
        }
    }

    #[test]
    fn generate_tensor_deterministic() {
        let shape = pmg_core::Shape::new(vec![5, 5]).unwrap();
        let plan1 = GenerationPlan::new("tensor", shape.clone(), pmg_core::DType::F32, 42).unwrap();
        let plan2 = GenerationPlan::new("tensor", shape, pmg_core::DType::F32, 42).unwrap();
        let mut rng1 = MockRng::new(42);
        let mut rng2 = MockRng::new(42);
        let values1 = generate_tensor(&plan1, &mut rng1).unwrap();
        let values2 = generate_tensor(&plan2, &mut rng2).unwrap();
        assert_eq!(values1, values2);
    }

    #[test]
    fn generate_tensor_different_seeds() {
        let shape = pmg_core::Shape::new(vec![5, 5]).unwrap();
        let plan1 = GenerationPlan::new("tensor", shape.clone(), pmg_core::DType::F32, 42).unwrap();
        let plan2 = GenerationPlan::new("tensor", shape, pmg_core::DType::F32, 43).unwrap();
        let mut rng1 = MockRng::new(42);
        let mut rng2 = MockRng::new(43);
        let values1 = generate_tensor(&plan1, &mut rng1).unwrap();
        let values2 = generate_tensor(&plan2, &mut rng2).unwrap();
        assert_ne!(values1, values2);
    }

    #[test]
    fn generated_tensor_to_bytes_f32() {
        let values = vec![1.0, 2.0, 3.0];
        let tensor = GeneratedTensor::new("test", values);
        let bytes = tensor.to_bytes(pmg_core::DType::F32).unwrap();
        assert_eq!(bytes.len(), 3 * 4); // 3 floats × 4 octets
    }

    #[test]
    fn generated_tensor_to_bytes_unsupported() {
        let values = vec![1.0];
        let tensor = GeneratedTensor::new("test", values);
        let result = tensor.to_bytes(pmg_core::DType::Bool);
        assert!(result.is_err());
    }
}
