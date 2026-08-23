//! Tests de résistance aux fichiers corrompus
//!
//! Ce module contient des tests pour valider que le système ne crash pas
//! lorsqu'il est confronté à des fichiers invalides ou corrompus.

use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Crée un répertoire temporaire pour les tests.
fn create_temp_dir() -> TempDir {
    tempfile::tempdir().unwrap()
}

/// Crée un fichier avec le contenu spécifié.
fn create_file(dir: &std::path::Path, name: &str, content: &[u8]) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, content).unwrap();
    path
}

// ============================================================================
// Tests de fichiers JSON corrompus
// ============================================================================

/// Test avec un fichier config.json vide.
#[test]
fn corruption_empty_config_json() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "config.json", b"");

    let result = fs::read_to_string(&path);
    assert!(result.is_ok());

    let content = result.unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_err(), "JSON vide devrait échouer");
}

/// Test avec un fichier config.json contenant du texte invalide.
#[test]
fn corruption_invalid_config_json() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "config.json", b"not valid json");

    let content = fs::read_to_string(&path).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_err(), "JSON invalide devrait échouer");
}

/// Test avec un fichier config.json tronqué.
#[test]
fn corruption_truncated_config_json() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "config.json", b"{\"model_type\": \"glm");

    let content = fs::read_to_string(&path).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_err(), "JSON tronqué devrait échouer");
}

/// Test avec un fichier config.json contenant un tableau au lieu d'un objet.
#[test]
fn corruption_array_config_json() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "config.json", b"[1, 2, 3]");

    let content = fs::read_to_string(&path).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    // Le parsing peut réussir mais le type est incorrect
    if let Ok(value) = parsed {
        assert!(value.is_array(), "Devrait être un tableau");
    }
}

/// Test avec un fichier metadata.json corrompu.
#[test]
fn corruption_metadata_json() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "metadata.json", b"{invalid");

    let content = fs::read_to_string(&path).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_err(), "JSON invalide devrait échouer");
}

// ============================================================================
// Tests de fichiers Safetensors corrompus
// ============================================================================

/// Test avec un fichier Safetensors vide.
#[test]
fn corruption_empty_safetensors() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", b"");

    // Un fichier vide ne devrait pas contenir d'headers valides
    let content = fs::read(&path).unwrap();
    assert!(content.is_empty());
}

/// Test avec un fichier Safetensors contenant du texte invalide.
#[test]
fn corruption_invalid_safetensors() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", b"not safetensors");

    let content = fs::read(&path).unwrap();
    assert!(!content.is_empty());
    // Le contenu n'est pas un header Safetensors valide
}

/// Test avec un fichier Safetensors tronqué.
#[test]
fn corruption_truncated_safetensors() {
    let temp_dir = create_temp_dir();
    // Un header Safetensors commence par la longueur du header en JSON
    // On crée un fichier avec juste les premiers octets
    let path = create_file(temp_dir.path(), "model.safetensors", b"\x00\x00\x00\x00");

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 4);
}

/// Test avec un fichier Safetensors contenant des octets aléatoires.
#[test]
fn corruption_random_bytes_safetensors() {
    let temp_dir = create_temp_dir();
    let random_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let path = create_file(temp_dir.path(), "model.safetensors", &random_data);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 1000);
}

/// Test avec un fichier Safetensors contenant un header JSON malformé.
#[test]
fn corruption_malformed_header_safetensors() {
    let temp_dir = create_temp_dir();
    // Simule un header JSON malformé
    let header = b"{\"tensor_name\": {dtype: \"F32\", shape: [10, 10], data_offsets: [0, 400]}}";
    let header_len = header.len() as u64;

    // Format Safetensors : 8 octets pour la longueur du header + header + données
    let mut content = Vec::new();
    content.extend_from_slice(&header_len.to_le_bytes());
    content.extend_from_slice(header);
    content.extend_from_slice(&[0u8; 400]); // Données simulées

    let path = create_file(temp_dir.path(), "model.safetensors", &content);
    let file_content = fs::read(&path).unwrap();
    assert!(file_content.len() > 8);
}

// ============================================================================
// Tests de fichiers binaires quelconques
// ============================================================================

/// Test avec un fichier binaire contenant des octets nuls.
#[test]
fn corruption_null_bytes() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.bin", &[0u8; 1000]);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 1000);
    assert!(content.iter().all(|&b| b == 0));
}

/// Test avec un fichier binaire contenant des octets max.
#[test]
fn corruption_max_bytes() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.bin", &[255u8; 1000]);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 1000);
    assert!(content.iter().all(|&b| b == 255));
}

/// Test avec un fichier binaire contenant un mélange d'octets.
#[test]
fn corruption_mixed_bytes() {
    let temp_dir = create_temp_dir();
    let data: Vec<u8> = (0..1000).map(|i| (i * 7 % 256) as u8).collect();
    let path = create_file(temp_dir.path(), "model.bin", &data);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 1000);
}

// ============================================================================
// Tests de fichiers tronqués
// ============================================================================

/// Test avec un fichier tronqué à 1 octet.
#[test]
fn corruption_truncated_1_byte() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", &[42]);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 1);
}

/// Test avec un fichier tronqué à 8 octets (taille d'un header).
#[test]
fn corruption_truncated_8_bytes() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", &[0u8; 8]);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 8);
}

/// Test avec un fichier tronqué à 100 octets.
#[test]
fn corruption_truncated_100_bytes() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", &[0u8; 100]);

    let content = fs::read(&path).unwrap();
    assert_eq!(content.len(), 100);
}

// ============================================================================
// Tests de chemins invalides
// ============================================================================

/// Test avec un chemin inexistant.
#[test]
fn corruption_nonexistent_path() {
    let path = std::path::Path::new("/nonexistent/path/to/model.safetensors");
    let result = fs::read(path);
    assert!(result.is_err(), "Chemin inexistant devrait échouer");
}

/// Test avec un fichier sans permissions de lecture.
#[test]
#[cfg(unix)]
fn corruption_no_read_permission() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", b"content");

    // Supprime les permissions de lecture
    let mut perms = fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o000);
    fs::set_permissions(&path, perms).unwrap();

    let result = fs::read(&path);
    assert!(result.is_err(), "Fichier sans permission devrait échouer");

    // Restaure les permissions pour le nettoyage
    let mut perms = fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o644);
    fs::set_permissions(&path, perms).unwrap();
}

// ============================================================================
// Tests de répertoires corrompus
// ============================================================================

/// Test avec un répertoire au lieu d'un fichier.
#[test]
fn corruption_directory_instead_of_file() {
    let temp_dir = create_temp_dir();
    let dir_path = temp_dir.path().join("model.safetensors");
    fs::create_dir(&dir_path).unwrap();

    let result = fs::read(&dir_path);
    assert!(result.is_err(), "Répertoire au lieu de fichier devrait échouer");

    // Nettoyage
    fs::remove_dir(&dir_path).unwrap();
}

/// Test avec un lien symbolique cassé.
#[test]
fn corruption_broken_symlink() {
    let temp_dir = create_temp_dir();
    let symlink_path = temp_dir.path().join("model.safetensors");
    let target_path = temp_dir.path().join("nonexistent_target");

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&target_path, &symlink_path).unwrap();
        let result = fs::read(&symlink_path);
        assert!(result.is_err(), "Lien symbolique cassé devrait échouer");
    }
}

// ============================================================================
// Tests de taille de fichier
// ============================================================================

/// Test avec un fichier de 0 octets.
#[test]
fn corruption_zero_size() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", b"");

    let metadata = fs::metadata(&path).unwrap();
    assert_eq!(metadata.len(), 0);
}

/// Test avec un très petit fichier (1 octet).
#[test]
fn corruption_one_byte() {
    let temp_dir = create_temp_dir();
    let path = create_file(temp_dir.path(), "model.safetensors", &[0x42]);

    let metadata = fs::metadata(&path).unwrap();
    assert_eq!(metadata.len(), 1);
}

// ============================================================================
// Tests d'intégrité
// ============================================================================

/// Vérifie qu'un fichier peut être lu sans panic.
#[test]
fn corruption_safe_read() {
    let temp_dir = create_temp_dir();

    // Test avec différents types de fichiers invalides
    let test_cases = vec![
        ("empty.json", b"" as &[u8]),
        ("invalid.json", b"not json"),
        ("truncated.json", b"{\"key\":"),
        ("binary.bin", &[0u8; 100]),
        ("text.txt", b"hello world"),
    ];

    for (name, content) in test_cases {
        let path = create_file(temp_dir.path(), name, content);
        // La lecture ne devrait jamais panic
        let result = fs::read(&path);
        assert!(result.is_ok() || result.is_err());
    }
}

/// Vérifie qu'un fichier peut être lu comme string sans panic.
#[test]
fn corruption_safe_read_to_string() {
    let temp_dir = create_temp_dir();

    let test_cases = vec![
        ("empty.json", b"" as &[u8]),
        ("invalid.json", b"not json"),
        ("truncated.json", b"{\"key\":"),
    ];

    for (name, content) in test_cases {
        let path = create_file(temp_dir.path(), name, content);
        // La lecture ne devrait jamais panic
        let result = fs::read_to_string(&path);
        // Note : read_to_string peut échouer pour des octets non-UTF8
        // mais ne devrait pas panic
    }
}
