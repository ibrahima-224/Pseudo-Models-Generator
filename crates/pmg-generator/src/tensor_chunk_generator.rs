//! # Générateur de chunks de tenseurs pour écriture directe sur disque
//!
//! Ce module implémente `TensorChunkGenerator`, un générateur qui produit
//! les chunks de tenseurs et les écrit directement sur disque via `ChunkWriter`
//! sans jamais charger le tenseur complet en mémoire. C'est le composant
//! central de l'optimisation mémoire de la phase 2.
//!
//! ## Objectifs
//! - Réduire la consommation mémoire de O(model_size) à O(chunk_size)
//! - Maintenir le déterminisme de la génération
//! - Intégrer la surveillance mémoire en temps réel
//! - Fournir des métriques de performance détaillées

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_io::pool::{PoolConfig, UnifiedBufferPool};
use pmg_io::safetensors::{ChunkWriterMetrics, ShardWriter};
use pmg_math::rng::DeterministicRng;

use crate::error::{GeneratorError, GeneratorResult};
use crate::memory_monitor::MemoryMonitor;
use crate::streaming_config::StreamingConfig;

/// Convertit un `pmg_core::DType` en `pmg_io::safetensors::DType`.
///
/// # Paramètres
/// - `core_dtype` : type de données du cœur
///
/// # Retourne
/// Le type de données correspondant pour Safetensors.
fn convert_dtype(core_dtype: &pmg_core::DType) -> pmg_io::safetensors::DType {
    match core_dtype {
        pmg_core::DType::F32 => pmg_io::safetensors::DType::F32,
        pmg_core::DType::F16 => pmg_io::safetensors::DType::F16,
        pmg_core::DType::Bf16 => pmg_io::safetensors::DType::BF16,
        pmg_core::DType::F8E4M3 => pmg_io::safetensors::DType::F8E4M3,
        pmg_core::DType::F8E5M2 => pmg_io::safetensors::DType::F8E5M2,
        pmg_core::DType::I8 => pmg_io::safetensors::DType::I8,
        pmg_core::DType::I16 => pmg_io::safetensors::DType::I16,
        pmg_core::DType::I32 => pmg_io::safetensors::DType::I32,
        pmg_core::DType::I64 => pmg_io::safetensors::DType::I64,
        pmg_core::DType::U8 => pmg_io::safetensors::DType::U8,
        pmg_core::DType::U16 => pmg_io::safetensors::DType::U16,
        pmg_core::DType::U32 => pmg_io::safetensors::DType::U32,
        pmg_core::DType::U64 => pmg_io::safetensors::DType::U64,
        pmg_core::DType::Bool => pmg_io::safetensors::DType::Bool,
        _ => pmg_io::safetensors::DType::F32, // Par défaut
    }
}

/// Générateur de chunks de tenseurs avec écriture directe sur disque.
///
/// Ce générateur produit les chunks d'un tenseur un par un et les écrit
/// directement sur disque via `ChunkWriter`. Il ne garde qu'un seul chunk
/// en mémoire à la fois, ce qui réduit la consommation mémoire à O(chunk_size).
pub struct TensorChunkGenerator {
    /// Configuration du streaming.
    config: StreamingConfig,
    /// Moniteur de mémoire pour surveiller la consommation.
    memory_monitor: MemoryMonitor,
    /// Taille des chunks en octets.
    chunk_size: usize,
    /// Seed de base pour la génération déterministe.
    base_seed: u64,
    /// Pool de buffers unifié pour optimiser les allocations (optionnel).
    buffer_pool: Option<UnifiedBufferPool>,
}

/// Résultat de la génération et écriture d'un tenseur.
#[derive(Debug, Clone)]
pub struct TensorGenerationResult {
    /// Nombre total d'éléments dans le tenseur.
    pub total_elements: usize,
    /// Nombre de chunks écrits.
    pub chunks_written: usize,
    /// Taille totale en octets écrite sur disque.
    pub total_bytes_written: usize,
    /// Métriques du writer.
    pub writer_metrics: ChunkWriterMetrics,
}

impl TensorChunkGenerator {
    /// Crée un nouveau générateur avec la configuration spécifiée.
    ///
    /// # Paramètres
    /// - `config` : configuration du streaming
    /// - `base_seed` : seed de base pour la génération déterministe
    ///
    /// # Retourne
    /// Un générateur prêt à l'emploi.
    pub fn new(config: StreamingConfig, base_seed: u64) -> Self {
        let memory_monitor = MemoryMonitor::new(config.max_memory);
        let chunk_size = config.chunk_size;

        // Initialiser le pool de buffers si configuré
        let buffer_pool = if config.use_buffer_pool {
            let pool_config = config.buffer_pool_config.clone().unwrap_or_else(|| {
                PoolConfig::new(
                    config.max_pool_memory,
                    1024 * 1024,      // 1 Mo taille min
                    64 * 1024 * 1024, // 64 Mo taille max
                    true,
                )
            });
            Some(UnifiedBufferPool::new(pool_config))
        } else {
            None
        };

        Self {
            config,
            memory_monitor,
            chunk_size,
            base_seed,
            buffer_pool,
        }
    }

    /// Génère et écrit un tenseur complet sur disque en chunks.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur à générer
    /// - `writer` : writer SafeTensors pour l'écriture sur disque
    /// - `tensor_index` : index du tenseur dans le modèle (pour le déterminisme)
    ///
    /// # Retourne
    /// Le résultat de la génération avec les métriques.
    ///
    /// # Erreurs
    /// Retourne une erreur si la génération ou l'écriture échoue.
    pub fn generate_and_write_tensor(
        &mut self,
        tensor_spec: &TensorSpec,
        writer: &mut ShardWriter,
        tensor_index: usize,
    ) -> GeneratorResult<TensorGenerationResult> {
        // Calculer le nombre total d'éléments
        let total_elements = tensor_spec.num_elements()? as usize;

        // Calculer la taille totale en octets selon le type de données
        let element_size = self.element_size(&tensor_spec.dtype);
        let _total_bytes = total_elements * element_size;

        // Vérifier que la mémoire est suffisante pour au moins un chunk
        let chunk_elements = self.chunk_size / element_size;
        if chunk_elements == 0 {
            return Err(GeneratorError::Internal(
                "La taille du chunk est trop petite pour un seul élément".to_string(),
            ));
        }

        // Allouer la mémoire pour le monitoring
        if !self.memory_monitor.allocate(self.chunk_size as u64) {
            return Err(GeneratorError::Internal(
                "Mémoire insuffisante pour allouer un chunk".to_string(),
            ));
        }

        // Générer et écrire les chunks un par un
        let mut total_chunks = 0;
        let mut total_bytes_written = 0;
        let mut offset = 0;

        while offset < total_elements {
            // Calculer la taille de ce chunk
            let current_chunk_elements = (total_elements - offset).min(chunk_elements);
            let current_chunk_bytes = current_chunk_elements * element_size;

            // Générer le chunk de valeurs
            let chunk_values = self.generate_chunk_values(
                tensor_spec,
                offset,
                current_chunk_elements,
                tensor_index,
            )?;

            // Convertir les valeurs en bytes
            let chunk_bytes = self.values_to_bytes(&chunk_values, &tensor_spec.dtype);

            // Écrire le chunk sur disque
            // Si c'est le premier chunk, on commence le tenseur
            if offset == 0 {
                let io_dtype = convert_dtype(&tensor_spec.dtype);
                writer
                    .begin_tensor(&tensor_spec.name, io_dtype, tensor_spec.shape.dims())
                    .map_err(|e| {
                        GeneratorError::Internal(format!("erreur début tenseur : {}", e))
                    })?;
            }

            writer
                .write_chunk(&chunk_bytes)
                .map_err(|e| GeneratorError::Internal(format!("erreur écriture chunk : {}", e)))?;

            // Si c'est le dernier chunk, on termine le tenseur
            if offset + current_chunk_elements >= total_elements {
                writer
                    .end_tensor()
                    .map_err(|e| GeneratorError::Internal(format!("erreur fin tenseur : {}", e)))?;
            }

            // Mettre à jour les compteurs
            total_chunks += 1;
            total_bytes_written += current_chunk_bytes;
            offset += current_chunk_elements;
        }

        // Libérer la mémoire allouée pour le monitoring
        self.memory_monitor.release(self.chunk_size as u64);

        // Retourner le résultat
        // Note: ShardWriter n'a pas de méthode metrics(), on retourne des métriques vides
        Ok(TensorGenerationResult {
            total_elements,
            chunks_written: total_chunks,
            total_bytes_written,
            writer_metrics: ChunkWriterMetrics::new(),
        })
    }

    /// Génère les valeurs pour un chunk d'un tenseur.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `offset` : offset du premier élément du chunk
    /// - `chunk_size` : nombre d'éléments dans le chunk
    /// - `tensor_index` : index du tenseur (pour le déterminisme)
    ///
    /// # Retourne
    /// Un vecteur de valeurs f64 générées.
    fn generate_chunk_values(
        &self,
        _tensor_spec: &TensorSpec,
        offset: usize,
        chunk_size: usize,
        tensor_index: usize,
    ) -> GeneratorResult<Vec<f64>> {
        // Dériver un seed spécifique pour ce chunk
        let chunk_seed = self.derive_chunk_seed(tensor_index, offset);

        // Créer un RNG déterministe
        let mut rng = DeterministicRng::from_seed(derive_seed_from_u64(chunk_seed));

        // Générer les valeurs selon la distribution normale (Box-Muller)
        let mut values = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            values.push(z);
        }

        Ok(values)
    }

    /// Convertit des valeurs f64 en bytes selon le type de données.
    ///
    /// # Paramètres
    /// - `values` : valeurs à convertir
    /// - `dtype` : type de données cible
    ///
    /// # Retourne
    /// Un vecteur de bytes représentant les valeurs.
    fn values_to_bytes(&self, values: &[f64], dtype: &pmg_core::DType) -> Vec<u8> {
        // Calculer la taille nécessaire en octets
        let element_size = self.element_size(dtype);
        let total_size = values.len() * element_size;

        // Utiliser le pool de buffers si disponible
        let mut bytes = if let Some(ref pool) = self.buffer_pool {
            pool.acquire_u8(total_size)
        } else {
            Vec::with_capacity(total_size)
        };

        match dtype {
            pmg_core::DType::F32 => {
                for &value in values {
                    bytes.extend_from_slice(&(value as f32).to_le_bytes());
                }
            },
            pmg_core::DType::F16 => {
                // Conversion en f16 IEEE 754
                for &value in values {
                    let f16_value = half::f16::from_f32(value as f32);
                    bytes.extend_from_slice(&f16_value.to_le_bytes());
                }
            },
            pmg_core::DType::Bf16 => {
                // Conversion en BFloat16
                for &value in values {
                    let f32_val = value as f32;
                    let bytes_val = f32_val.to_le_bytes();
                    // BFloat16 = les 16 bits de poids fort du f32
                    bytes.push(bytes_val[2]);
                    bytes.push(bytes_val[3]);
                }
            },
            pmg_core::DType::I64 => {
                for &value in values {
                    bytes.extend_from_slice(&(value as i64).to_le_bytes());
                }
            },
            pmg_core::DType::I32 => {
                for &value in values {
                    bytes.extend_from_slice(&(value as i32).to_le_bytes());
                }
            },
            pmg_core::DType::I16 => {
                for &value in values {
                    bytes.extend_from_slice(&(value as i16).to_le_bytes());
                }
            },
            pmg_core::DType::I8 => {
                for &value in values {
                    bytes.push(value as i8 as u8);
                }
            },
            // Pour les autres types, on convertit en f32 par défaut
            _ => {
                for &value in values {
                    bytes.extend_from_slice(&(value as f32).to_le_bytes());
                }
            },
        }

        bytes
    }

    /// Génère un chunk de valeurs et les convertit directement en bytes.
    ///
    /// Cette méthode fusionne la génération et la conversion pour éviter
    /// l'allocation intermédiaire d'un `Vec<f64>`. Les valeurs sont générées
    /// et converties directement en format bytes (little-endian).
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `offset` : offset en éléments depuis le début du tenseur
    /// - `chunk_size` : nombre d'éléments à générer
    /// - `tensor_index` : index du tenseur dans le plan
    ///
    /// # Retourne
    /// Un vecteur de bytes contenant les valeurs générées.
    #[allow(dead_code)]
    fn generate_chunk_as_bytes(
        &self,
        _tensor_spec: &TensorSpec,
        offset: usize,
        chunk_size: usize,
        tensor_index: usize,
    ) -> GeneratorResult<Vec<u8>> {
        // Calculer la taille en octets
        let element_size = self.element_size(&_tensor_spec.dtype);
        let total_bytes = chunk_size * element_size;

        // Allouer le buffer de bytes
        let mut bytes = if let Some(ref pool) = self.buffer_pool {
            pool.acquire_u8(total_bytes)
        } else {
            Vec::with_capacity(total_bytes)
        };

        // Dériver la seed pour ce chunk (même logique que generate_chunk_values)
        let chunk_seed = self.derive_chunk_seed(tensor_index, offset);
        let mut rng = DeterministicRng::from_seed(derive_seed_from_u64(chunk_seed));

        // Générer les valeurs et convertir directement en bytes
        // Distribution normale via Box-Muller (même algorithme que generate_chunk_values)
        for _ in 0..chunk_size {
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // Conversion directe selon le dtype
            match _tensor_spec.dtype {
                pmg_core::DType::F32 => {
                    let f32_val = z as f32;
                    bytes.extend_from_slice(&f32_val.to_le_bytes());
                },
                pmg_core::DType::F16 => {
                    let f16_val = half::f16::from_f32(z as f32);
                    bytes.extend_from_slice(&f16_val.to_le_bytes());
                },
                pmg_core::DType::Bf16 => {
                    let f32_val = z as f32;
                    let bytes_val = f32_val.to_le_bytes();
                    // BFloat16 = les 16 bits de poids fort du f32
                    bytes.push(bytes_val[2]);
                    bytes.push(bytes_val[3]);
                },
                pmg_core::DType::F64 => {
                    bytes.extend_from_slice(&z.to_le_bytes());
                },
                pmg_core::DType::I32 => {
                    let i32_val = z as i32;
                    bytes.extend_from_slice(&i32_val.to_le_bytes());
                },
                pmg_core::DType::I16 => {
                    let i16_val = z as i16;
                    bytes.extend_from_slice(&i16_val.to_le_bytes());
                },
                pmg_core::DType::I8 => {
                    bytes.push(z as i8 as u8);
                },
                _ => {
                    // Fallback : conversion en f32
                    let f32_val = z as f32;
                    bytes.extend_from_slice(&f32_val.to_le_bytes());
                },
            }
        }

        Ok(bytes)
    }

    /// Retourne la taille en octets d'un élément selon le type de données.
    ///
    /// # Paramètres
    /// - `dtype` : type de données
    ///
    /// # Retourne
    /// La taille en octets d'un élément.
    fn element_size(&self, dtype: &pmg_core::DType) -> usize {
        match dtype {
            pmg_core::DType::F32 | pmg_core::DType::I32 | pmg_core::DType::U32 => 4,
            pmg_core::DType::F16
            | pmg_core::DType::Bf16
            | pmg_core::DType::I16
            | pmg_core::DType::U16 => 2,
            pmg_core::DType::I64 | pmg_core::DType::U64 | pmg_core::DType::F64 => 8,
            pmg_core::DType::I8 | pmg_core::DType::U8 | pmg_core::DType::Bool => 1,
            // Pour les autres types, on utilise la taille par défaut (4 octets)
            _ => 4,
        }
    }

    /// Dérive un seed spécifique pour un chunk donné.
    ///
    /// # Paramètres
    /// - `tensor_index` : index du tenseur
    /// - `chunk_offset` : offset du chunk dans le tenseur
    ///
    /// # Retourne
    /// Un seed dérivé unique pour ce chunk.
    fn derive_chunk_seed(&self, tensor_index: usize, chunk_offset: usize) -> u64 {
        // Mélange déterministe pour garantir des seeds uniques
        self.base_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(tensor_index as u64)
            .wrapping_add(chunk_offset as u64)
    }

    /// Retourne une référence au moniteur de mémoire.
    pub fn memory_monitor(&self) -> &MemoryMonitor {
        &self.memory_monitor
    }

    /// Retourne la configuration du streaming.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Fonction utilitaire pour dériver un seed à partir d'un u64.
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

// ============================================================================
// Tests unitaires
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::{DType, Shape, TensorRole};
    use tempfile::tempdir;

    /// Test de création du générateur
    #[test]
    fn test_tensor_chunk_generator_creation() {
        let config = StreamingConfig::default();
        let generator = TensorChunkGenerator::new(config, 42);

        assert_eq!(generator.chunk_size, 8 * 1024 * 1024);
        assert_eq!(generator.base_seed, 42);
    }

    /// Test de dérivation de seed
    #[test]
    fn test_derive_chunk_seed() {
        let config = StreamingConfig::default();
        let generator = TensorChunkGenerator::new(config, 42);

        let seed1 = generator.derive_chunk_seed(0, 0);
        let seed2 = generator.derive_chunk_seed(0, 1);
        let seed3 = generator.derive_chunk_seed(1, 0);

        // Les seeds doivent être différents pour des chunks différents
        assert_ne!(seed1, seed2);
        // Les seeds doivent être différents pour des tenseurs différents
        assert_ne!(seed1, seed3);
    }

    /// Test de taille d'élément
    #[test]
    fn test_element_size() {
        let config = StreamingConfig::default();
        let generator = TensorChunkGenerator::new(config, 42);

        assert_eq!(generator.element_size(&DType::F32), 4);
        assert_eq!(generator.element_size(&DType::F16), 2);
        assert_eq!(generator.element_size(&DType::I64), 8);
        assert_eq!(generator.element_size(&DType::I8), 1);
    }

    /// Test de conversion de valeurs en bytes
    #[test]
    fn test_values_to_bytes() {
        let config = StreamingConfig::default();
        let generator = TensorChunkGenerator::new(config, 42);

        let values = vec![1.0, 2.0, 3.0];
        let bytes = generator.values_to_bytes(&values, &DType::F32);

        assert_eq!(bytes.len(), 12); // 3 * 4 octets
                                     // Vérifier la conversion little-endian
        let val1 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let val2 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let val3 = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert!((val1 - 1.0).abs() < f32::EPSILON);
        assert!((val2 - 2.0).abs() < f32::EPSILON);
        assert!((val3 - 3.0).abs() < f32::EPSILON);
    }

    /// Test de génération et écriture d'un tenseur
    #[test]
    fn test_generate_and_write_tensor() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let config = StreamingConfig::new(1024 * 1024, 10 * 1024 * 1024); // 1 Mo chunks, 10 Mo max
        let mut generator = TensorChunkGenerator::new(config, 42);

        // Créer un tenseur de test
        let tensor_spec = TensorSpec::new(
            "test_tensor",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap();

        // Créer le writer ShardWriter avec une réserve d'en-tête de 1 Ko
        let mut writer = ShardWriter::new(path, 1024).unwrap();

        // Générer et écrire le tenseur
        let result = generator
            .generate_and_write_tensor(&tensor_spec, &mut writer, 0)
            .unwrap();

        // Vérifications
        assert_eq!(result.total_elements, 6400); // 100 * 64
        assert!(result.chunks_written >= 1);
        assert!(result.total_bytes_written > 0);

        // Finaliser le writer
        writer.finalize().unwrap();
    }

    /// Test du monitoring mémoire
    #[test]
    fn test_memory_monitoring() {
        let config = StreamingConfig::new(1024 * 1024, 10 * 1024 * 1024); // 1 Mo chunks, 10 Mo max
        let generator = TensorChunkGenerator::new(config, 42);

        // Vérifier que le moniteur est initialisé
        assert_eq!(generator.memory_monitor().usage_percentage(), 0.0);
        assert!(!generator.memory_monitor().is_near_limit());
    }
}
