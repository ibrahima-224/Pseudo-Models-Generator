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

//! Tests unitaires pour le module report.

use super::*;
use crate::inspector::{InspectionLevel, InspectionReport};
use std::path::PathBuf;

#[test]
fn test_structured_report_creation() {
    let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Normal);
    let structured = StructuredReport::from_inspection_report(&report);

    assert_eq!(structured.model_path, PathBuf::from("/fake/model"));
    assert_eq!(structured.level, "Normal");
    assert!(structured.timestamp.is_none());
}

#[test]
fn test_structured_report_json_serialization() {
    let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Normal);
    let structured = StructuredReport::from_inspection_report(&report);

    let json = structured.to_json();
    assert!(serde_json::from_str::<StructuredReport>(&json).is_ok());
    assert!(json.contains("model_path"));
    assert!(json.contains("/fake/model"));
}

#[test]
fn test_structured_report_compact_json() {
    let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Verbose);
    let structured = StructuredReport::from_inspection_report(&report);

    let json_compact = structured.to_json_compact();
    assert!(serde_json::from_str::<StructuredReport>(&json_compact).is_ok());
    // Le JSON compact ne doit pas contenir de sauts de ligne inutiles
    assert!(!json_compact.contains("\n  "));
}

#[test]
fn test_structured_report_text_levels() {
    let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Normal);
    let structured = StructuredReport::from_inspection_report(&report);

    let brief = structured.to_text(InspectionLevel::Brief);
    assert!(brief.contains("=== Inspection bref du modèle ==="));

    let normal = structured.to_text(InspectionLevel::Normal);
    assert!(normal.contains("=== Rapport d'inspection du modèle ==="));

    let verbose = structured.to_text(InspectionLevel::Verbose);
    assert!(verbose.contains("=== Rapport d'inspection du modèle ==="));
    assert!(verbose.contains("--- Détails Safetensors ---"));

    let debug = structured.to_text(InspectionLevel::Debug);
    assert!(debug.contains("--- Informations de débogage ---"));
}

#[test]
fn test_format_number() {
    assert_eq!(format_number(0), "0");
    assert_eq!(format_number(123), "123");
    assert_eq!(format_number(1234), "1_234");
    assert_eq!(format_number(1234567), "1_234_567");
    assert_eq!(format_number(1234567890), "1_234_567_890");
}

#[test]
fn test_format_bytes() {
    assert_eq!(format_bytes(0), "0 o");
    assert_eq!(format_bytes(1023), "1023 o");
    assert_eq!(format_bytes(1024), "1.00 KiB");
    assert_eq!(format_bytes(1048576), "1.00 MiB");
    assert_eq!(format_bytes(1073741824), "1.00 GiB");
    assert_eq!(format_bytes(1099511627776), "1.00 TiB");
}

#[test]
fn test_with_timestamp() {
    let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Normal);
    let structured = StructuredReport::from_inspection_report(&report)
        .with_timestamp("2024-01-01T00:00:00Z".to_string());

    assert_eq!(
        structured.timestamp,
        Some("2024-01-01T00:00:00Z".to_string())
    );
}
