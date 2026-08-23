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

//! Tests unitaires pour le module `manifest`.

use super::*;

#[test]
fn test_new_default() {
    let metadata = PmgMetadata::new_default();
    assert_eq!(metadata.format, "pmg-metadata");
    assert_eq!(metadata.format_version, 1);
    assert!(metadata.synthetic);
    assert!(metadata.validate().is_ok());
}

#[test]
fn test_validate_invalid_format() {
    let mut metadata = PmgMetadata::new_default();
    metadata.format = "invalid".to_string();
    assert!(metadata.validate().is_err());
}

#[test]
fn test_validate_invalid_version() {
    let mut metadata = PmgMetadata::new_default();
    metadata.format_version = 2;
    assert!(metadata.validate().is_err());
}

#[test]
fn test_validate_missing_hash_prefix() {
    let mut metadata = PmgMetadata::new_default();
    metadata.source_metadata_hash = "00000000".to_string();
    assert!(metadata.validate().is_err());
}

#[test]
fn test_validate_zero_size() {
    let mut metadata = PmgMetadata::new_default();
    metadata.actual_size_bytes = 0;
    assert!(metadata.validate().is_err());
}

#[test]
fn test_json_roundtrip() {
    let metadata = PmgMetadata::new_default();
    let json = metadata.to_json().unwrap();
    let deserialized = PmgMetadata::from_json(&json).unwrap();
    assert_eq!(metadata, deserialized);
}

#[test]
fn test_display_french() {
    let metadata = PmgMetadata::new_default();
    let display = metadata.display_french();
    assert!(display.contains("Manifeste PMG"));
    assert!(display.contains("glm-5.2"));
    assert!(display.contains("bf16"));
}

#[test]
fn test_retrocompatibility_pseudo_model() {
    let mut metadata = PmgMetadata::new_default();
    metadata.pseudo_model = Some("old-model".to_string());
    let json = metadata.to_json().unwrap();
    assert!(json.contains("pseudo_model"));
    let deserialized = PmgMetadata::from_json(&json).unwrap();
    assert_eq!(deserialized.pseudo_model, Some("old-model".to_string()));
}

#[test]
fn test_retrocompatibility_weights_are_synthetic() {
    let mut metadata = PmgMetadata::new_default();
    metadata.weights_are_synthetic = Some(true);
    let json = metadata.to_json().unwrap();
    assert!(json.contains("weights_are_synthetic"));
    let deserialized = PmgMetadata::from_json(&json).unwrap();
    assert_eq!(deserialized.weights_are_synthetic, Some(true));
}
