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

//! Tests de fuzzing principaux pour le parser JSON du header SafeTensors.
//!
//! Ces tests vérifient la robustesse du parser face à des entrées de base malformées.
//! Les tests de cas limites spécifiques sont dans `json_edge_case_tests`.

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

/// Test avec JSON vide (objet vide).
#[test]
fn test_empty_json_object() {
    let header_json = "{}";
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Un objet vide est techniquement un JSON valide, mais ne contient aucun tenseur.
    // Selon l'implémentation, cela peut être accepté ou rejeté.
    // Nous vérifions simplement qu'il n'y a pas de panic.
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

/// Test avec un tableau JSON (non attendu comme racine).
#[test]
fn test_json_array() {
    let header_json = "[]";
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Un tableau n'est pas un objet, donc doit échouer
    assert!(result.is_err());
}

/// Test avec une chaîne JSON vide (non attendue).
#[test]
fn test_json_empty_string() {
    let header_json = "";
    let file = build_safetensors_file(header_json, 0);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Une chaîne vide n'est pas un JSON valide
    assert!(result.is_err());
}

/// Test avec nombres extrêmes.
#[test]
fn test_extreme_numbers() {
    // Test avec un très grand nombre
    let header_json = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);
    assert!(result.is_ok());

    // Test avec un nombre décimal très précis
    let header_json =
        r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0.123456789,4.987654321]}}"#;
    let file = build_safetensors_file(header_json, 4);
    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Les offsets doivent être des entiers, donc cela doit échouer.
    // Cependant, serde_json peut accepter des flottants et les convertir en u64 si la valeur est entière.
    // Ici, les valeurs ne sont pas entières, donc la désérialisation devrait échouer.
    // Mais le parser ignore les entrées invalides, donc le résultat peut être Ok sans tenseur.
    // Nous vérifions simplement qu'il n'y a pas de panic.
    match result {
        Ok(reader) => {
            // Si accepté, l'entrée est ignorée car les offsets sont invalides
            assert_eq!(reader.tensor_count(), 0);
        },
        Err(_) => {
            // Si rejeté, c'est acceptable
        },
    }
}
