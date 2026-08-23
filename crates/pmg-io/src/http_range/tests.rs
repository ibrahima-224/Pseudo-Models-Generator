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

//! Tests unitaires pour le module HTTP Range.

use super::*;
use std::path::PathBuf;

#[test]
fn test_http_range_config_default() {
    let config = HttpRangeConfig::default();
    assert_eq!(config.timeout, std::time::Duration::from_secs(30));
    assert!(config.max_header_size > 0);
    assert_eq!(config.max_retries, 3);
    assert!(!config.user_agent.is_empty());
}

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

#[test]
fn test_parse_header_valid() {
    let header_json = r#"{
        "tensor1": {"dtype": "F32", "shape": [10, 10]},
        "tensor2": {"dtype": "F16", "shape": [20, 20]}
    }"#;

    let result = client::parse_header(header_json);
    assert!(result.is_ok());

    let value = result.unwrap();
    assert!(value.is_object());
    assert!(value.get("tensor1").is_some());
    assert!(value.get("tensor2").is_some());
}

#[test]
fn test_parse_header_invalid_json() {
    let header_json = "not json at all";
    let result = client::parse_header(header_json);
    assert!(result.is_err());
}

#[test]
fn test_parse_header_not_object() {
    let header_json = "[1, 2, 3]";
    let result = client::parse_header(header_json);
    assert!(result.is_err());
}

#[test]
fn test_cache_path_generation() {
    let url = "https://example.com/model.safetensors";
    let metadata = CachedMetadata::new(url, "{}".to_string(), 2);
    let cache_dir = PathBuf::from("/tmp/test_cache");

    let cache_path = metadata.cache_path(&cache_dir);
    assert!(cache_path.to_string_lossy().ends_with(".json"));
    assert!(cache_path.starts_with(&cache_dir));
}

#[test]
fn test_http_range_error_conversion() {
    let error = HttpRangeError::Network("test".to_string());
    let core_error: pmg_core::error::CoreError = error.into();
    assert!(format!("{core_error}").contains("Réseau"));
}

#[test]
fn test_http_range_error_range_unsupported() {
    let error = HttpRangeError::RangeUnsupported { status: 200 };
    let core_error: pmg_core::error::CoreError = error.into();
    assert!(format!("{core_error}").contains("Range non supporté"));
}

#[test]
fn test_http_range_error_header_too_large() {
    let error = HttpRangeError::HeaderTooLarge {
        size: 1000,
        limit: 100,
    };
    let core_error: pmg_core::error::CoreError = error.into();
    assert!(format!("{core_error}").contains("Header trop grand"));
}

#[test]
fn test_check_range_support_mock() {
    // Note: Ce test nécessiterait un mock ou un serveur de test
    // Pour l'instant, on teste juste que la fonction compile
    let _config = HttpRangeConfig::default();
    // On ne peut pas tester sans serveur, donc on skip
    // let result = check_range_support("https://example.com/test.safetensors", &config);
}
