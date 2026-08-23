//! # Configuration du mode streaming
//!
//! Ce module fournit la configuration pour le mode streaming de génération
//! de modèles, permettant de réduire la consommation mémoire à O(chunk_size).

use std::fmt;

/// Configuration du mode streaming.
///
/// Cette structure définit les paramètres pour le mode streaming de génération,
/// permettant de contrôler la taille des chunks, la mémoire maximale et
/// d'autres options liées à l'optimisation mémoire.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Taille des chunks en octets (défaut : 8 Mo).
    pub chunk_size: usize,
    /// Mémoire maximale autorisée en octets (défaut : 500 Mo).
    pub max_memory: u64,
    /// Activer le pool de buffers (défaut : true).
    pub use_buffer_pool: bool,
    /// Taille maximale du pool en octets (défaut : 32 Mo).
    pub max_pool_memory: usize,
    /// Configuration du pool de buffers unifié (optionnel).
    /// Si `Some`, utilise le pool unifié au lieu des paramètres séparés.
    pub buffer_pool_config: Option<pmg_io::pool::PoolConfig>,
    /// Activer la synchronisation disque après chaque tenseur (défaut : false).
    pub sync_after_tensor: bool,
    /// Mode verbose (défaut : false).
    pub verbose: bool,
}

impl Default for StreamingConfig {
    /// Crée une configuration par défaut.
    ///
    /// # Retourne
    /// Une configuration avec les valeurs par défaut :
    /// - chunk_size : 8 Mo
    /// - max_memory : 500 Mo
    /// - use_buffer_pool : true
    /// - max_pool_memory : 32 Mo
    /// - sync_after_tensor : false
    /// - verbose : false
    fn default() -> Self {
        Self {
            chunk_size: 8 * 1024 * 1024,   // 8 Mo
            max_memory: 500 * 1024 * 1024, // 500 Mo
            use_buffer_pool: true,
            max_pool_memory: 32 * 1024 * 1024, // 32 Mo
            buffer_pool_config: None,
            sync_after_tensor: false,
            verbose: false,
        }
    }
}

impl StreamingConfig {
    /// Crée une nouvelle configuration avec des paramètres personnalisés.
    ///
    /// # Paramètres
    /// - `chunk_size` : taille des chunks en octets
    /// - `max_memory` : mémoire maximale autorisée en octets
    ///
    /// # Retourne
    /// Une configuration avec les paramètres spécifiés et les autres valeurs par défaut.
    pub fn new(chunk_size: usize, max_memory: u64) -> Self {
        Self {
            chunk_size,
            max_memory,
            ..Default::default()
        }
    }

    /// Vérifie si la configuration est valide.
    ///
    /// # Retourne
    /// `true` si la configuration est valide, `false` sinon.
    pub fn is_valid(&self) -> bool {
        // Vérification des tailles minimales et maximales
        self.chunk_size >= 1024 * 1024 && // Au moins 1 Mo
        self.chunk_size <= 64 * 1024 * 1024 && // Au plus 64 Mo
        self.max_memory >= 100 * 1024 * 1024 && // Au moins 100 Mo
        self.max_memory <= 16 * 1024 * 1024 * 1024 // Au plus 16 Go
    }

    /// Convertit la taille des chunks en mégaoctets.
    ///
    /// # Retourne
    /// La taille des chunks en Mo.
    pub fn chunk_size_mb(&self) -> f64 {
        self.chunk_size as f64 / (1024.0 * 1024.0)
    }

    /// Convertit la mémoire maximale en mégaoctets.
    ///
    /// # Retourne
    /// La mémoire maximale en Mo.
    pub fn max_memory_mb(&self) -> f64 {
        self.max_memory as f64 / (1024.0 * 1024.0)
    }
}

impl fmt::Display for StreamingConfig {
    /// Affiche la configuration de manière lisible.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StreamingConfig {{ chunk_size: {:.2} Mo, max_memory: {:.2} Mo, use_buffer_pool: {}, max_pool_memory: {:.2} Mo, sync_after_tensor: {}, verbose: {} }}",
            self.chunk_size_mb(),
            self.max_memory_mb(),
            self.use_buffer_pool,
            self.max_pool_memory as f64 / (1024.0 * 1024.0),
            self.sync_after_tensor,
            self.verbose
        )
    }
}

// ============================================================================
// Tests unitaires
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test de création d'une configuration par défaut
    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();

        assert_eq!(config.chunk_size, 8 * 1024 * 1024);
        assert_eq!(config.max_memory, 500 * 1024 * 1024);
        assert!(config.use_buffer_pool);
        assert_eq!(config.max_pool_memory, 32 * 1024 * 1024);
        assert!(!config.sync_after_tensor);
        assert!(!config.verbose);
    }

    /// Test de création d'une configuration personnalisée
    #[test]
    fn test_streaming_config_new() {
        let config = StreamingConfig::new(16 * 1024 * 1024, 1024 * 1024 * 1024);

        assert_eq!(config.chunk_size, 16 * 1024 * 1024);
        assert_eq!(config.max_memory, 1024 * 1024 * 1024);
        // Les autres valeurs doivent être par défaut
        assert!(config.use_buffer_pool);
        assert_eq!(config.max_pool_memory, 32 * 1024 * 1024);
    }

    /// Test de validation de configuration
    #[test]
    fn test_streaming_config_is_valid() {
        // Configuration valide
        let valid_config = StreamingConfig::default();
        assert!(valid_config.is_valid());

        // Configuration avec chunk trop petit
        let invalid_config1 = StreamingConfig {
            chunk_size: 512 * 1024, // 512 Ko < 1 Mo
            ..Default::default()
        };
        assert!(!invalid_config1.is_valid());

        // Configuration avec mémoire trop faible
        let invalid_config2 = StreamingConfig {
            max_memory: 50 * 1024 * 1024, // 50 Mo < 100 Mo
            ..Default::default()
        };
        assert!(!invalid_config2.is_valid());
    }

    /// Test des conversions en Mo
    #[test]
    fn test_streaming_config_conversions() {
        let config = StreamingConfig::default();

        assert!((config.chunk_size_mb() - 8.0).abs() < f64::EPSILON);
        assert!((config.max_memory_mb() - 500.0).abs() < f64::EPSILON);
    }

    /// Test de l'affichage
    #[test]
    fn test_streaming_config_display() {
        let config = StreamingConfig::default();
        let display = format!("{}", config);

        assert!(display.contains("chunk_size"));
        assert!(display.contains("max_memory"));
        assert!(display.contains("use_buffer_pool"));
    }
}
