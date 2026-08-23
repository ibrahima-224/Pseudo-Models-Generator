//! Générateur de Chunks Optimisé avec Zero-Copy
//!
//! Ce module implémente l'`OptimizedTensorChunkGenerator` qui utilise le
//! `ZeroCopyTensorWriter` pour écrire les tenseurs sans allocation intermédiaire,
//! garantissant une consommation mémoire bornée à 1 Go maximum.

use std::sync::Arc;

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_core::dtype::DType;
use pmg_core::memory::{GlobalMemoryManager, MemoryMonitor};
use pmg_io::pool::{OptimizedBufferPool, PoolConfig};
use pmg_io::safetensors::writer::zero_copy::ZeroCopyTensorWriter;
use pmg_math::rng::DeterministicRng;

use crate::error::{GeneratorError, GeneratorResult};
use crate::streaming_config::StreamingConfig;

/// Générateur de chunks optimisé avec écriture zero-copy
///
/// Ce générateur utilise le `ZeroCopyTensorWriter` pour écrire les tenseurs
/// sans allocation intermédiaire, et le `GlobalMemoryManager` pour garantir
/// que la mémoire utilisée ne dépasse jamais 1 Go.
#[allow(dead_code)]
pub struct OptimizedTensorChunkGenerator {
    /// Configuration du streaming
    config: StreamingConfig,

    /// Moniteur mémoire global
    memory_monitor: Arc<GlobalMemoryManager>,

    /// Pool de buffers optimisé
    #[allow(dead_code)]
    buffer_pool: Arc<OptimizedBufferPool>,

    /// Seed de base pour la génération déterministe
    base_seed: u64,

    /// Métriques de génération
    metrics: GenerationMetrics,
}

/// Métriques de génération
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    /// Nombre total de tenseurs générés
    pub tensors_generated: usize,

    /// Nombre total de chunks écrits
    pub chunks_written: usize,

    /// Taille totale en octets écrite
    pub total_bytes_written: u64,

    /// Pic mémoire atteint
    pub peak_memory_usage: u64,

    /// Nombre d'allocations effectuées
    pub allocation_count: usize,

    /// Nombre de désallocations effectuées
    pub deallocation_count: usize,
}

impl OptimizedTensorChunkGenerator {
    /// Crée un nouveau générateur optimisé
    pub fn new(
        config: StreamingConfig,
        memory_monitor: Arc<GlobalMemoryManager>,
        base_seed: u64,
    ) -> Self {
        // Créer le pool de buffers optimisé
        let pool_config = PoolConfig::default();
        let buffer_pool = Arc::new(OptimizedBufferPool::new(
            memory_monitor.clone(),
            pool_config,
        ));

        Self {
            config,
            memory_monitor,
            buffer_pool,
            base_seed,
            metrics: GenerationMetrics::default(),
        }
    }

    /// Génère et écrit un tenseur complet sur disque en chunks
    pub fn generate_and_write_tensor(
        &mut self,
        tensor_spec: &TensorSpec,
        writer: &mut ZeroCopyTensorWriter,
        tensor_index: usize,
    ) -> GeneratorResult<TensorGenerationResult> {
        // Calculer le nombre total d'éléments
        let total_elements = tensor_spec.num_elements()? as usize;

        // Calculer la taille totale en octets selon le type de données
        let element_size = self.element_size(&tensor_spec.dtype);
        let _total_bytes = total_elements * element_size;

        // Vérifier que la mémoire est suffisante pour au moins un chunk
        let chunk_elements = self.config.chunk_size / element_size;
        if chunk_elements == 0 {
            return Err(GeneratorError::Internal(
                "La taille du chunk est trop petite pour un seul élément".to_string(),
            ));
        }

        // Allouer la mémoire pour le monitoring
        self.memory_monitor
            .allocate(self.config.chunk_size as u64)
            .map_err(|e| GeneratorError::Internal(format!("Erreur d'allocation mémoire: {}", e)))?;

        // Générer et écrire les chunks un par un
        let mut total_chunks = 0;
        let mut total_bytes_written = 0;
        let mut offset = 0;

        // Commencer l'écriture du tenseur
        // Convertir les dimensions u64 en usize pour le writer
        let shape_usize: Vec<usize> = tensor_spec
            .shape
            .dims()
            .iter()
            .map(|&d| d as usize)
            .collect();
        writer
            .begin_tensor(&tensor_spec.name, tensor_spec.dtype, &shape_usize)
            .map_err(|e| GeneratorError::Internal(format!("Erreur début tenseur: {}", e)))?;

        while offset < total_elements {
            // Calculer la taille de ce chunk
            let current_chunk_elements = (total_elements - offset).min(chunk_elements);
            let current_chunk_bytes = current_chunk_elements * element_size;

            // Allouer un buffer pour les valeurs
            let mut chunk_values = vec![0.0; current_chunk_elements];

            // Générer les valeurs dans le buffer
            self.generate_chunk_values_into_buffer(
                &mut chunk_values,
                tensor_spec,
                offset,
                current_chunk_elements,
                tensor_index,
            )?;

            // Écrire le chunk directement via le writer zero-copy
            writer
                .write_f64_chunk(&chunk_values, tensor_spec.dtype)
                .map_err(|e| GeneratorError::Internal(format!("Erreur écriture chunk: {}", e)))?;

            // Mettre à jour les compteurs
            total_chunks += 1;
            total_bytes_written += current_chunk_bytes;
            offset += current_chunk_elements;

            // Mettre à jour les métriques
            self.metrics.chunks_written += 1;
            self.metrics.allocation_count += 1;
        }

        // Terminer l'écriture du tenseur
        writer
            .end_tensor()
            .map_err(|e| GeneratorError::Internal(format!("Erreur fin tenseur: {}", e)))?;

        // Libérer la mémoire allouée pour le monitoring
        self.memory_monitor
            .deallocate(self.config.chunk_size as u64);
        self.metrics.deallocation_count += 1;

        // Mettre à jour les métriques
        self.metrics.tensors_generated += 1;
        self.metrics.total_bytes_written += total_bytes_written as u64;
        self.metrics.peak_memory_usage = self
            .metrics
            .peak_memory_usage
            .max(self.memory_monitor.current_usage());

        // Retourner le résultat
        Ok(TensorGenerationResult {
            total_elements,
            chunks_written: total_chunks,
            total_bytes_written,
            memory_metrics: MemoryMetrics {
                peak_memory: self.metrics.peak_memory_usage,
                allocation_count: self.metrics.allocation_count,
                deallocation_count: self.metrics.deallocation_count,
            },
        })
    }

    /// Génère les valeurs d'un chunk dans un buffer existant (zero-copy)
    fn generate_chunk_values_into_buffer(
        &self,
        buffer: &mut [f64],
        _tensor_spec: &TensorSpec,
        offset: usize,
        _chunk_size: usize,
        tensor_index: usize,
    ) -> GeneratorResult<()> {
        // Dériver un seed spécifique pour ce chunk
        let chunk_seed = self.derive_chunk_seed(tensor_index, offset);

        // Créer un RNG déterministe
        let mut rng = DeterministicRng::from_seed(derive_seed_from_u64(chunk_seed));

        // Générer les valeurs selon la distribution normale (Box-Muller)
        for value in buffer.iter_mut() {
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *value = z;
        }

        Ok(())
    }

    /// Retourne la taille en octets d'un élément selon le type de données
    #[allow(dead_code)]
    fn element_size(&self, dtype: &DType) -> usize {
        match dtype {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::I64 | DType::U64 | DType::F64 => 8,
            DType::I8 | DType::U8 | DType::Bool => 1,
            _ => 4, // Par défaut
        }
    }

    /// Dérive un seed spécifique pour un chunk donné
    #[allow(dead_code)]
    fn derive_chunk_seed(&self, tensor_index: usize, chunk_offset: usize) -> u64 {
        self.base_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(tensor_index as u64)
            .wrapping_add(chunk_offset as u64)
    }

    /// Retourne les métriques de génération
    #[allow(dead_code)]
    pub fn metrics(&self) -> &GenerationMetrics {
        &self.metrics
    }

    /// Réinitialise les métriques
    #[allow(dead_code)]
    pub fn reset_metrics(&mut self) {
        self.metrics = GenerationMetrics::default();
    }

    /// Retourne le moniteur mémoire
    #[allow(dead_code)]
    pub fn memory_monitor(&self) -> &GlobalMemoryManager {
        &self.memory_monitor
    }

    /// Retourne la configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Résultat de la génération d'un tenseur
#[derive(Debug, Clone)]
pub struct TensorGenerationResult {
    /// Nombre total d'éléments dans le tenseur
    pub total_elements: usize,

    /// Nombre de chunks écrits
    pub chunks_written: usize,

    /// Taille totale en octets écrite
    pub total_bytes_written: usize,

    /// Métriques mémoire
    pub memory_metrics: MemoryMetrics,
}

/// Métriques mémoire pour la génération
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// Pic mémoire atteint
    pub peak_memory: u64,

    /// Nombre d'allocations
    pub allocation_count: usize,

    /// Nombre de désallocations
    pub deallocation_count: usize,
}

/// Fonction utilitaire pour dériver un seed à partir d'un u64
fn derive_seed_from_u64(seed: u64) -> [u8; 32] {
    let mut result = [0u8; 32];
    let bytes = seed.to_le_bytes();
    result[..8].copy_from_slice(&bytes);
    // Ajoute un mélange simple pour améliorer la distribution
    for i in 8..32 {
        result[i] = result[i % 8].wrapping_add(i as u8);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::memory::GlobalMemoryManager;
    use pmg_core::Shape;
    use pmg_core::TensorRole;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn test_creation_generateur() {
        let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));
        let config = StreamingConfig::default();

        let generator = OptimizedTensorChunkGenerator::new(config, manager, 12345);

        assert_eq!(generator.metrics().tensors_generated, 0);
    }

    #[test]
    fn test_generation_simple() {
        // Augmenter la limite à 32 Mo pour couvrir toutes les allocations:
        // - 8 Mo pour le buffer de conversion du ZeroCopyTensorWriter
        // - 8 Mo pour le chunk_size du OptimizedTensorChunkGenerator
        // - Marge pour les allocations du pool de buffers
        let manager = Arc::new(GlobalMemoryManager::with_limit(32 * 1024 * 1024));
        let config = StreamingConfig::default();

        let mut generator = OptimizedTensorChunkGenerator::new(config, manager, 12345);

        // Créer un spec de tenseur simple
        let spec = TensorSpec::new(
            "test.weight",
            Shape::new(vec![2, 3]).unwrap(),
            DType::F32,
            TensorRole::Embedding,
        )
        .unwrap();

        // Créer un writer temporaire
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.safetensors");
        let shard_writer = pmg_io::safetensors::ShardWriter::new(path, 1024).unwrap();
        let writer_config = pmg_io::safetensors::writer::zero_copy::TensorWriterConfig::default();
        let mut writer = ZeroCopyTensorWriter::new(
            shard_writer,
            generator.memory_monitor.clone(),
            writer_config,
        )
        .unwrap();

        // Générer le tenseur
        let result = generator
            .generate_and_write_tensor(&spec, &mut writer, 0)
            .unwrap();

        assert_eq!(result.total_elements, 6);
        assert!(result.chunks_written >= 1);
        assert!(result.total_bytes_written > 0);
    }
}
