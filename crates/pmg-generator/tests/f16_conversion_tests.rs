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

//! Tests de conversion F16 pour le crate pmg-generator.
//!
//! Ces tests vérifient la conversion correcte des valeurs f64 vers le format
//! F16 (half-precision) via la crate `half`. La conversion passe par f32
//! comme étape intermédiaire pour gérer l'arrondi avant la troncation.

use pmg_core::DType;
use pmg_generator::GeneratedTensor;

/// Test : vérifie que la conversion F16 produit le bon nombre d'octets.
///
/// Chaque élément f16 occupe 2 octets. Ce test s'assure que la taille
/// du vecteur de bytes est correcte pour différents nombres d'éléments.
#[test]
fn test_tensor_to_bytes_f16_conversion() {
    // Données de test avec 5 éléments
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor = GeneratedTensor::new("test_tensor", values);

    // Conversion en F16
    let bytes = tensor.to_bytes(DType::F16).unwrap();

    // Vérifier que la taille est correcte : 5 éléments × 2 octets = 10 octets
    assert_eq!(
        bytes.len(),
        10,
        "La taille devrait être de 10 octets (5 × 2)"
    );

    // Vérifier que chaque paire d'octets est valide (pas de panic)
    for i in 0..5 {
        let offset = i * 2;
        let _chunk = &bytes[offset..offset + 2];
    }
}

/// Test : vérifie la précision de la conversion F16.
///
/// La conversion F16 a une précision limitée (environ 3 décimales).
/// Ce test vérifie que les valeurs reconnaissables restent correctes
/// après conversion.
#[test]
fn test_tensor_to_bytes_f16_accuracy() {
    // Valeurs testées : entiers simples et décimales
    let values = vec![0.0, 1.0, 2.5, 10.0, 100.0];
    let tensor = GeneratedTensor::new("accuracy_test", values.clone());

    let bytes = tensor.to_bytes(DType::F16).unwrap();

    // Reconvertir les octets en f16 pour vérifier la précision
    use half::f16;
    for (i, &expected) in values.iter().enumerate() {
        let offset = i * 2;
        let f16_bytes = [bytes[offset], bytes[offset + 1]];
        let f16_value = f16::from_le_bytes(f16_bytes);

        // La conversion f64 → f32 → f16 peut introduire une erreur d'arrondi
        // On vérifie que la valeur est proche de l'attendu (avec une tolérance)
        let f32_value = expected as f32;
        let f16_expected = f16::from_f32(f32_value);

        // Comparaison via les bits pour éviter les problèmes de NaN
        assert_eq!(
            f16_value.to_bits(),
            f16_expected.to_bits(),
            "La valeur F16 devrait correspondre pour l'élément {}",
            i
        );
    }
}

/// Test : vérifie le comportement avec un tableau vide.
///
/// Un tableau vide devrait produire un vecteur de bytes vide.
#[test]
fn test_tensor_to_bytes_f16_empty() {
    let values: Vec<f64> = vec![];
    let tensor = GeneratedTensor::new("empty_tensor", values);

    let bytes = tensor.to_bytes(DType::F16).unwrap();

    // Vérifier que le résultat est vide
    assert!(
        bytes.is_empty(),
        "Un tableau vide devrait produire 0 octets"
    );
}

/// Test : vérifie la conversion avec des valeurs spéciales (NaN, infini).
///
/// Ces valeurs doivent être correctement converties en F16 sans panic.
#[test]
fn test_tensor_to_bytes_f16_special_values() {
    let values = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let tensor = GeneratedTensor::new("special_values", values);

    // La conversion ne doit pas panic
    let bytes = tensor.to_bytes(DType::F16).unwrap();

    // Vérifier la taille
    assert_eq!(bytes.len(), 10, "5 éléments × 2 octets = 10 octets");

    // Vérifier que les octets sont valides (pas de panic lors de la relecture)
    use half::f16;
    for i in 0..5 {
        let offset = i * 2;
        let f16_bytes = [bytes[offset], bytes[offset + 1]];
        let _f16_value = f16::from_le_bytes(f16_bytes);
    }
}

/// Test : vérifie la reproductibilité de la conversion.
///
/// La même entrée devrait toujours produire la même sortie.
#[test]
fn test_tensor_to_bytes_f16_reproducibility() {
    let values = vec![1.23456, 7.89012, 3.45678];
    let tensor1 = GeneratedTensor::new("repro_test", values.clone());
    let tensor2 = GeneratedTensor::new("repro_test", values);

    let bytes1 = tensor1.to_bytes(DType::F16).unwrap();
    let bytes2 = tensor2.to_bytes(DType::F16).unwrap();

    assert_eq!(bytes1, bytes2, "La conversion devrait être déterministe");
}
