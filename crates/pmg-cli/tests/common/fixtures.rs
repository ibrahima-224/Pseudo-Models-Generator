//! Fixtures et données de test prédéfinies pour les tests de la CLI PMG.
//!
//! Ce module contient des constantes et des structures de données réutilisables
//! pour tous les tests.

/// Modèles supportés par la CLI PMG.
pub const SUPPORTED_MODELS: &[&str] = &["glm52", "deepseek_v4_flash"];

/// Modes de génération supportés.
pub const SUPPORTED_GENERATION_MODES: &[&str] = &["safe", "realistic"];

/// Formats de sortie supportés.
pub const SUPPORTED_OUTPUT_FORMATS: &[&str] = &["text", "json"];

/// Types de données supportés.
pub const SUPPORTED_DTYPES: &[&str] = &["f32", "f16", "bf16"];

/// Codes de sortie attendus.
#[allow(dead_code)]
pub mod exit_codes {
    /// Code de sortie pour succès.
    pub const SUCCESS: i32 = 0;
    /// Code de sortie pour erreur générale.
    pub const GENERAL_ERROR: i32 = 1;
    /// Code de sortie pour argument invalide.
    pub const INVALID_ARGUMENT: i32 = 2;
    /// Code de sortie pour modèle invalide.
    pub const INVALID_MODEL: i32 = 3;
    /// Code de sortie pour erreur I/O.
    pub const IO_ERROR: i32 = 4;
    /// Code de sortie pour validation échouée.
    pub const VALIDATION_FAILED: i32 = 5;
    /// Code de sortie pour comparaison incompatible.
    pub const INCOMPATIBLE_COMPARISON: i32 = 6;
}

/// Tailles de test prédéfinies en octets.
#[allow(dead_code)]
pub mod sizes {
    /// 1 octet (taille minimale invalide).
    pub const ONE_BYTE: &str = "1B";
    /// 1 kilooctet.
    pub const ONE_KB: &str = "1K";
    /// 1 mégaoctet.
    pub const ONE_MB: &str = "1M";
    /// 100 mégaoctets.
    pub const ONE_HUNDRED_MB: &str = "100M";
    /// 500 mégaoctets.
    pub const FIVE_HUNDRED_MB: &str = "500M";
    /// 1 gigaoctet.
    pub const ONE_GB: &str = "1G";
    /// 10 gigaoctets (grande taille).
    pub const TEN_GB: &str = "10G";
    /// 100 gigaoctets (taille extrême).
    pub const ONE_HUNDRED_GB: &str = "100G";
}

/// Seeds de test prédéfinies.
#[allow(dead_code)]
pub mod seeds {
    /// Seed standard pour les tests.
    pub const STANDARD: u64 = 42;
    /// Seed alternative pour les tests de reproductibilité.
    pub const ALTERNATIVE: u64 = 43;
    /// Seed minimale.
    pub const MINIMAL: u64 = 0;
    /// Seed maximale (u64::MAX).
    pub const MAXIMAL: u64 = u64::MAX;
}

/// Noms de fichiers de test.
#[allow(dead_code)]
pub mod filenames {
    /// Nom du fichier de modèle de test.
    pub const TEST_MODEL: &str = "test_model";
    /// Nom du fichier de configuration de test.
    pub const TEST_CONFIG: &str = "test_config";
    /// Nom du fichier de sortie de test.
    pub const TEST_OUTPUT: &str = "test_output";
    /// Nom du fichier de rapport de test.
    pub const TEST_REPORT: &str = "test_report";
}

/// Messages d'erreur attendus.
#[allow(dead_code)]
pub mod error_messages {
    /// Message d'erreur pour modèle non supporté.
    pub const INVALID_MODEL: &str = "Modèle non supporté";
    /// Message d'erreur pour fichier introuvable.
    pub const FILE_NOT_FOUND: &str = "fichier introuvable";
    /// Message d'erreur pour permission refusée.
    pub const PERMISSION_DENIED: &str = "Permission refusée";
    /// Message d'erreur pour chemin invalide.
    pub const INVALID_PATH: &str = "Chemin invalide";
    /// Message d'erreur pour taille invalide.
    pub const INVALID_SIZE: &str = "Taille invalide";
    /// Message d'erreur pour format non supporté.
    pub const INVALID_FORMAT: &str = "Format non supporté";
    /// Message d'erreur pour tolérance invalide.
    pub const INVALID_TOLERANCE: &str = "Tolérance invalide";
}

/// Chemins de test pour les tests de path traversal.
pub mod traversal_paths {
    /// Chemin relatif suspect pour path traversal.
    pub const SUSPICIOUS_RELATIVE: &str = "../../etc/passwd";
    /// Chemin absolu sensible.
    pub const SENSITIVE_ABSOLUTE: &str = "/etc/passwd";
    /// Chemin avec séquence de path traversal.
    pub const TRAVERSAL_SEQUENCE: &str = "test/../../../etc/shadow";
    /// Chemin avec backslashes (Windows).
    pub const BACKSLASH_PATH: &str = "..\\..\\etc\\passwd";
}

/// Arguments d'injection pour les tests de sécurité.
pub mod injection_args {
    /// Injection de commande shell.
    pub const SHELL_INJECTION: &str = "glm52; rm -rf /";
    /// Injection de pipe.
    pub const PIPE_INJECTION: &str = "glm52 | cat /etc/passwd";
    /// Injection de backticks.
    pub const BACKTICK_INJECTION: &str = "glm52 `whoami`";
    /// Injection de dollar parentheses.
    pub const DOLLAR_PAREN_INJECTION: &str = "glm52 $(whoami)";
    /// Injection de caractères spéciaux.
    pub const SPECIAL_CHARS: &str = "glm52\"; echo hacked;\"";
    /// Injection de unicode.
    pub const UNICODE_INJECTION: &str = "glm52\u{200B}injection";
}
