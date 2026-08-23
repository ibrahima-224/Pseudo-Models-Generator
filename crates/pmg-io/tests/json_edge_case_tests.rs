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

//! Tests de cas limites pour le parser JSON du header SafeTensors.
//!
//! Ces tests vérifient la robustesse du parser face à des entrées malformées
//! spécifiques, incluant des cas limites comme JSON vide, structures profondément
//! imbriquées, strings très longues, nombres extrêmes, caractères spéciaux, etc.

use pmg_io::safetensors::read_header_from;
use std::io::Cursor;

/// Construit un fichier SafeTensors minimal avec le header JSON spécifié.
///
/// # Paramètres
///
/// * `header_json` - Le contenu JSON du header (sans padding).
/// * `payload_size` - Taille du payload fictif en octets.
///
/// # Retour
///
/// Un vecteur d'octets représentant un fichier SafeTensors valide.
fn build_safetensors_file(header_json: &str, payload_size: usize) -> Vec<u8> {
    let json_len = header_json.len();
    let padding = (8 - (json_len % 8)) % 8;
    let padded_json = format!("{}{}", header_json, " ".repeat(padding));
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&vec![0u8; payload_size]);
    file
}

/// Test avec JSON profondément imbriqué (1000 niveaux).
#[test]
fn test_deeply_nested_json() {
    // Construit un objet imbriqué 1000 fois : {"a":{"b":{"c":...}}}
    let mut nested = String::from("{");
    for _ in 0..999 {
        nested.push_str("\"a\":{");
    }
    nested.push_str("\"a\":1");
    for _ in 0..999 {
        nested.push('}');
    }
    nested.push('}');

    let file = build_safetensors_file(&nested, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le parser doit gérer sans stack overflow ou erreur interne
    match result {
        Ok(reader) => {
            // Si accepté, il ne doit pas y avoir de tenseurs valides
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Si rejeté, c'est acceptable (profondeur trop grande)
        },
    }
}

/// Test avec strings très longues (1 Mo de caractères).
#[test]
fn test_very_long_strings() {
    // Crée une valeur de 1 Mo de caractères
    let long_value = "x".repeat(1024 * 1024);
    let header_json = format!("{{\"key\":\"{}\"}}", long_value);

    let file = build_safetensors_file(&header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le parser doit gérer sans déborder de mémoire
    match result {
        Ok(reader) => {
            // Si accepté, il ne doit pas y avoir de tenseurs
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Si rejeté, c'est acceptable
        },
    }
}

/// Test avec caractères spéciaux dans les noms de clés.
#[test]
fn test_special_characters_in_keys() {
    // Noms de clés avec caractères spéciaux (caractères de contrôle)
    // Les caractères de contrôle peuvent être échappés dans les chaînes JSON.
    let header_json = r#"{"\u0000":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le parser peut accepter ou rejeter ces caractères.
    // Nous vérifions simplement qu'il n'y a pas de panic.
    match result {
        Ok(reader) => {
            // Si accepté, vérifier que la clé est correctement parsée
            assert!(reader.header.contains_key("\u{0}"));
        },
        Err(_) => {
            // Si rejeté, c'est acceptable
        },
    }
}

/// Test avec newlines et tabs dans les noms.
#[test]
fn test_newlines_and_tabs() {
    let header_json = r#"{"na\nme":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Les newlines dans les noms sont échappés en JSON, donc le parser peut les accepter.
    // Nous vérifions simplement qu'il n'y a pas de panic.
    match result {
        Ok(reader) => {
            // Si accepté, vérifier que la clé contient un newline
            assert!(reader.header.contains_key("na\nme"));
        },
        Err(_) => {
            // Si rejeté, c'est acceptable
        },
    }
}

/// Test avec caractères UTF-8 invalides.
#[test]
fn test_invalid_utf8() {
    // Construit un buffer avec des octets invalides UTF-8
    let mut header_buf = Vec::new();
    // Ajoute des octets valides d'abord
    header_buf.extend_from_slice(b"{\"key\":\"");
    // Ajoute des octets invalides UTF-8 (octet de continuation sans début)
    header_buf.extend_from_slice(&[0x80, 0x80, 0x80]);
    header_buf.extend_from_slice(b"\"}");

    let padding = (8 - (header_buf.len() % 8)) % 8;
    header_buf.extend_from_slice(&vec![b' '; padding]);

    let header_size = header_buf.len() as u64;
    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(&header_buf);
    file.extend_from_slice(&[0u8; 10]);

    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);
    assert!(result.is_err());
}

/// Test avec JSON tronqué (sans fermante).
#[test]
fn test_truncated_json() {
    let header_json = r#"{"name": "test""#; // Manque la fermante }
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);
    assert!(result.is_err());
}

/// Test avec double quotes dans les noms de clés.
#[test]
fn test_double_quotes_in_keys() {
    // Les noms de clés avec des quotes internes doivent être échappées
    let header_json = r#"{"\"name\"":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Cela peut être accepté ou rejeté selon l'implémentation
    match result {
        Ok(reader) => {
            // Si accepté, vérifier que la clé est correctement parsée
            assert!(reader.header.contains_key("\"name\""));
        },
        Err(_) => {
            // Si rejeté, c'est acceptable
        },
    }
}

/// Test avec keys vides.
#[test]
fn test_empty_keys() {
    let header_json = r#"{"": {"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Une clé vide peut être acceptée ou rejetée
    match result {
        Ok(reader) => {
            assert!(reader.header.contains_key(""));
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec values nulles (null).
#[test]
fn test_null_values() {
    let header_json = r#"{"tensor": null}"#;
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // null n'est pas un TensorHeaderEntry valide, donc l'entrée est ignorée.
    // Le parser peut réussir sans tenseur.
    match result {
        Ok(reader) => {
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec arrays vides pour shape.
#[test]
fn test_empty_shape() {
    let header_json = r#"{"tensor":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Une forme vide peut être acceptée (scalaire) ou rejetée
    match result {
        Ok(reader) => {
            let entry = reader.header.get("tensor").unwrap();
            assert!(entry.shape.is_empty());
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec header trop grand (dépasse MAX_HEADER_SIZE).
#[test]
fn test_oversized_header() {
    // Crée un JSON valide mais déclare une taille trop grande
    let header_json = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_oversized_header(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le parser doit détecter que la taille déclarée est trop grande par rapport au contenu réel.
    // Cependant, notre construction crée un fichier avec une taille déclarée plus grande que le contenu.
    // Le parser va essayer de lire header_size octets, mais il n'y en a pas assez.
    // Cela devrait provoquer une erreur I/O (UnexpectedEof) ou une erreur HeaderTooLarge.
    // Nous vérifions simplement qu'il y a une erreur.
    assert!(result.is_err());
}

/// Test avec offsets négatifs (non supportés en Rust, mais test via JSON).
#[test]
fn test_negative_offsets() {
    let header_json = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[-1,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Les offsets négatifs ne sont pas autorisés en u64.
    // La désérialisation devrait échouer, donc l'entrée est ignorée.
    match result {
        Ok(reader) => {
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec offsets en doubles.
#[test]
fn test_float_offsets() {
    let header_json = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0.5,4.5]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Les offsets doivent être des entiers.
    // La désérialisation devrait échouer, donc l'entrée est ignorée.
    match result {
        Ok(reader) => {
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec dtype invalide.
#[test]
fn test_invalid_dtype() {
    let header_json = r#"{"tensor":{"dtype":"INVALID","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le dtype invalide provoque une erreur de désérialisation, donc l'entrée est ignorée.
    match result {
        Ok(reader) => {
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec shape contenant des chaînes.
#[test]
fn test_shape_with_strings() {
    let header_json = r#"{"tensor":{"dtype":"F32","shape":["a","b"],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // La forme doit être un tableau d'entiers.
    // La désérialisation devrait échouer, donc l'entrée est ignorée.
    match result {
        Ok(reader) => {
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Acceptable
        },
    }
}

/// Test avec header contenant des champs supplémentaires.
#[test]
fn test_extra_fields() {
    let header_json =
        r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":"field"}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Les champs supplémentaires doivent être ignorés
    assert!(result.is_ok());
    let reader = result.unwrap();
    let entry = reader.header.get("tensor").unwrap();
    assert_eq!(entry.dtype, pmg_io::safetensors::DType::F32);
}

/// Test avec un seul caractère nul dans une clé.
#[test]
fn test_single_null_byte() {
    let header_json = r#"{"\u0000":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le caractère nul peut être accepté ou rejeté
    if let Ok(reader) = result {
        assert!(reader.header.contains_key("\u{0}"));
    }
}

/// Test avec un header qui n'est pas du JSON du tout.
#[test]
fn test_non_json_content() {
    let header_json = "This is not JSON at all!";
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);
    assert!(result.is_err());
}

/// Test avec un header qui est du JSON valide mais pas un objet.
#[test]
fn test_json_number() {
    let header_json = "12345";
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);
    assert!(result.is_err());
}

/// Test avec un header qui est du JSON valide mais pas un objet.
#[test]
fn test_json_boolean() {
    let header_json = "true";
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);
    assert!(result.is_err());
}

/// Construit un fichier SafeTensors avec un header qui dépasse la taille maximale.
fn build_oversized_header(header_json: &str, payload_size: usize) -> Vec<u8> {
    let padding = (8 - (header_json.len() % 8)) % 8;
    let padded_json = format!("{}{}", header_json, " ".repeat(padding));
    // Déclarer une taille plus grande que le contenu réel pour simuler un header trop grand
    let header_size = padded_json.len() as u64 + 1024;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&vec![0u8; payload_size]);
    file
}
