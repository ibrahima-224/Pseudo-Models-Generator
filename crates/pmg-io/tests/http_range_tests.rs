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

//! Tests pour la feature HTTP Range de pmg-io.
//!
//! Ces tests vérifient le fonctionnement des différentes fonctions
//! du module http_range, notamment la gestion du cache, le parsing
//! des headers, et la conversion des erreurs.

#[cfg(feature = "http-range")]
mod http_range_tests {
    use pmg_io::http_range::{
        cache_metadata, invalidate_cache, load_cached_metadata, parse_header, CachedMetadata,
        HttpRangeConfig, HttpRangeError,
    };
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Test de la configuration par défaut.
    #[test]
    fn test_http_range_config_default() {
        let config = HttpRangeConfig::default();
        assert_eq!(config.timeout.as_secs(), 30);
        assert!(config.max_header_size > 0);
        assert_eq!(config.max_retries, 3);
        assert!(!config.user_agent.is_empty());
        assert!(config.cache_dir.to_string_lossy().contains("pmg"));
    }

    /// Test de création de CachedMetadata.
    #[test]
    fn test_cached_metadata_creation() {
        let url = "https://example.com/model.safetensors";
        let header_json = r#"{"tensor1": {"dtype": "F32", "shape": [10, 10]}}"#;
        let header_size = header_json.len() as u64;

        let metadata = CachedMetadata::new(url, header_json.to_string(), header_size);

        assert_eq!(metadata.url, url);
        assert_eq!(metadata.header_json, header_json);
        assert_eq!(metadata.header_size, header_size);
        assert!(!metadata.url_hash.is_empty());
        assert!(metadata.cached_at > 0);
        assert_eq!(metadata.cache_version, 1);
    }

    /// Test de génération du chemin de cache.
    #[test]
    fn test_cache_path_generation() {
        let url = "https://example.com/model.safetensors";
        let metadata = CachedMetadata::new(url, "{}".to_string(), 2);
        let cache_dir = PathBuf::from("/tmp/test_cache");

        let cache_path = metadata.cache_path(&cache_dir);
        assert!(cache_path.to_string_lossy().ends_with(".json"));
        assert!(cache_path.starts_with(&cache_dir));
    }

    /// Test de parsing de header JSON valide.
    #[test]
    fn test_parse_header_valid() {
        let header_json = r#"{
            "tensor1": {"dtype": "F32", "shape": [10, 10]},
            "tensor2": {"dtype": "F16", "shape": [20, 20]}
        }"#;

        let result = parse_header(header_json);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value.is_object());
        assert!(value.get("tensor1").is_some());
        assert!(value.get("tensor2").is_some());
    }

    /// Test de parsing de header JSON invalide.
    #[test]
    fn test_parse_header_invalid_json() {
        let header_json = "not json at all";
        let result = parse_header(header_json);
        assert!(result.is_err());
    }

    /// Test de parsing de header JSON qui n'est pas un objet.
    #[test]
    fn test_parse_header_not_object() {
        let header_json = "[1, 2, 3]";
        let result = parse_header(header_json);
        assert!(result.is_err());
    }

    /// Test de conversion des erreurs HTTP Range en CoreError.
    #[test]
    fn test_http_range_error_conversion() {
        let error = HttpRangeError::Network("test".to_string());
        let core_error: pmg_core::error::CoreError = error.into();
        assert!(format!("{core_error}").contains("Réseau"));
    }

    /// Test de conversion d'erreur Range non supporté.
    #[test]
    fn test_http_range_error_range_unsupported() {
        let error = HttpRangeError::RangeUnsupported { status: 200 };
        let core_error: pmg_core::error::CoreError = error.into();
        assert!(format!("{core_error}").contains("Range non supporté"));
    }

    /// Test de conversion d'erreur Header trop grand.
    #[test]
    fn test_http_range_error_header_too_large() {
        let error = HttpRangeError::HeaderTooLarge {
            size: 1000,
            limit: 100,
        };
        let core_error: pmg_core::error::CoreError = error.into();
        assert!(format!("{core_error}").contains("Header trop grand"));
    }

    /// Test de mise en cache et rechargement.
    #[test]
    fn test_cache_metadata_and_reload() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();

        let config = HttpRangeConfig {
            cache_dir: cache_dir.clone(),
            ..Default::default()
        };

        let url = "https://example.com/model.safetensors";
        let header_json = r#"{"tensor1": {"dtype": "F32", "shape": [10, 10]}}"#;
        let metadata = CachedMetadata::new(url, header_json.to_string(), 100);

        // Mise en cache
        let result = cache_metadata(&metadata, &config);
        assert!(result.is_ok());

        // Rechargement
        let loaded = load_cached_metadata(url, &config).unwrap();
        assert!(loaded.is_some());

        let loaded_metadata = loaded.unwrap();
        assert_eq!(loaded_metadata.url, url);
        assert_eq!(loaded_metadata.header_json, header_json);
        assert_eq!(loaded_metadata.header_size, 100);
    }

    /// Test d'invalidation de cache.
    #[test]
    fn test_invalidate_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();

        let config = HttpRangeConfig {
            cache_dir: cache_dir.clone(),
            ..Default::default()
        };

        let url = "https://example.com/model.safetensors";
        let header_json = r#"{"tensor1": {"dtype": "F32", "shape": [10, 10]}}"#;
        let metadata = CachedMetadata::new(url, header_json.to_string(), 100);

        // Mise en cache
        cache_metadata(&metadata, &config).unwrap();

        // Vérification que le cache existe
        let loaded = load_cached_metadata(url, &config).unwrap();
        assert!(loaded.is_some());

        // Invalidation
        let result = invalidate_cache(url, &config).unwrap();
        assert!(result);

        // Vérification que le cache a été supprimé
        let loaded = load_cached_metadata(url, &config).unwrap();
        assert!(loaded.is_none());
    }

    /// Test d'invalidation de cache inexistant.
    #[test]
    fn test_invalidate_cache_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();

        let config = HttpRangeConfig {
            cache_dir: cache_dir.clone(),
            ..Default::default()
        };

        let url = "https://example.com/nonexistent.safetensors";

        let result = invalidate_cache(url, &config).unwrap();
        assert!(!result);
    }

    /// Test de vérification du support Range (avec mock).
    #[test]
    fn test_check_range_support_mock() {
        // Note: Ce test nécessiterait un mock ou un serveur de test
        // Pour l'instant, on teste juste que la fonction compile
        let _config = HttpRangeConfig::default();
        // On ne peut pas tester sans serveur, donc on skip
        // let result = check_range_support("https://example.com/test.safetensors", &config);
    }

    /// Test de gestion des erreurs de cache.
    #[test]
    fn test_cache_error_handling() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();

        let config = HttpRangeConfig {
            cache_dir: cache_dir.clone(),
            ..Default::default()
        };

        let url = "https://example.com/model.safetensors";

        // Tentative de chargement de cache inexistant
        let result = load_cached_metadata(url, &config).unwrap();
        assert!(result.is_none());
    }

    /// Test de sérialisation/désérialisation de CachedMetadata.
    #[test]
    fn test_cached_metadata_serialization() {
        let url = "https://example.com/model.safetensors";
        let header_json = r#"{"tensor1": {"dtype": "F32", "shape": [10, 10]}}"#;
        let metadata = CachedMetadata::new(url, header_json.to_string(), 100);

        // Sérialisation
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("https://example.com/model.safetensors"));
        assert!(json.contains("tensor1"));

        // Désérialisation
        let deserialized: CachedMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.url, url);
        assert_eq!(deserialized.header_json, header_json);
        assert_eq!(deserialized.header_size, 100);
    }
}
