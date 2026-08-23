//! Kernels optimisés pour les shapes courantes
//!
//! Ce module fournit des kernels optimisés pour les shapes
//! les plus utilisées en machine learning.

use super::normal_generation::NormalGenerationAccelerated;
use crate::acceleration::GpuAccelerated;

/// Shapes courantes en ML
pub struct CommonShapes;

impl CommonShapes {
    /// Shape pour couches d'attention (768, 1024, 2048, 4096)
    pub const ATTENTION_DIMS: &[usize] = &[768, 1024, 2048, 4096];

    /// Shape pour couches feed-forward (3072, 4096, 8192, 16384)
    pub const FFN_DIMS: &[usize] = &[3072, 4096, 8192, 16384];

    /// Shape pour batch sizes (1, 8, 16, 32, 64)
    pub const BATCH_SIZES: &[usize] = &[1, 8, 16, 32, 64];

    /// Shape pour séquences (128, 256, 512, 1024, 2048)
    pub const SEQUENCE_LENGTHS: &[usize] = &[128, 256, 512, 1024, 2048];

    /// Vérifie si une shape est courante
    pub fn is_common_shape(size: usize) -> bool {
        Self::ATTENTION_DIMS.contains(&size)
            || Self::FFN_DIMS.contains(&size)
            || Self::BATCH_SIZES.contains(&size)
            || Self::SEQUENCE_LENGTHS.contains(&size)
    }
}

/// Générateur optimisé pour shapes courantes
pub struct OptimizedGenerator {
    /// Cache des générateurs par taille
    cache: std::collections::HashMap<usize, NormalGenerationAccelerated>,
}

impl OptimizedGenerator {
    /// Crée un nouveau générateur optimisé
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }

    /// Génère des données pour une shape courante
    pub fn generate_common_shape(
        &mut self,
        size: usize,
        mean: f64,
        std: f64,
        seed: u64,
    ) -> Vec<f64> {
        // Utiliser le cache si disponible
        if let Some(generator) = self.cache.get(&size) {
            return generator
                .execute_cpu(&(size, mean, std))
                .unwrap_or_default();
        }

        // Créer un nouveau générateur et le mettre en cache
        let generator = NormalGenerationAccelerated::new(seed);
        let result = generator
            .execute_cpu(&(size, mean, std))
            .unwrap_or_default();
        self.cache.insert(size, generator);
        result
    }

    /// Pré-charge les shapes courantes dans le cache
    pub fn preload_common_shapes(&mut self, seed: u64) {
        for &size in CommonShapes::ATTENTION_DIMS
            .iter()
            .chain(CommonShapes::FFN_DIMS.iter())
            .chain(CommonShapes::BATCH_SIZES.iter())
            .chain(CommonShapes::SEQUENCE_LENGTHS.iter())
        {
            let generator = NormalGenerationAccelerated::new(seed);
            self.cache.insert(size, generator);
        }
    }

    /// Retourne le nombre d'éléments en cache
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for OptimizedGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch generator pour traitement par lots
pub struct BatchGenerator {
    /// Taille du batch
    batch_size: usize,
    /// Générateur optimisé
    generator: OptimizedGenerator,
}

impl BatchGenerator {
    /// Crée un nouveau batch generator
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            generator: OptimizedGenerator::new(),
        }
    }

    /// Génère un batch complet
    pub fn generate_batch(
        &mut self,
        shape: usize,
        mean: f64,
        std: f64,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        (0..self.batch_size)
            .map(|i| {
                self.generator
                    .generate_common_shape(shape, mean, std, seed + i as u64)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_shapes() {
        assert!(CommonShapes::is_common_shape(768));
        assert!(CommonShapes::is_common_shape(1024));
        assert!(CommonShapes::is_common_shape(3072));
        assert!(!CommonShapes::is_common_shape(123));
    }

    #[test]
    fn test_optimized_generator() {
        let mut gen = OptimizedGenerator::new();
        let data = gen.generate_common_shape(768, 0.0, 1.0, 42);
        assert_eq!(data.len(), 768);
        assert!(gen.cache_size() > 0);
    }

    #[test]
    fn test_batch_generator() {
        let mut batch = BatchGenerator::new(8);
        let data = batch.generate_batch(768, 0.0, 1.0, 42);
        assert_eq!(data.len(), 8);
        assert_eq!(data[0].len(), 768);
    }
}
