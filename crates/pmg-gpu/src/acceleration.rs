//! Trait pour l'accélération GPU
//!
//! Ce module définit le trait `GpuAccelerated` pour les opérations
//! pouvant être accélérées par GPU. Il fournit une interface uniforme
//! pour l'exécution CPU/GPU avec fallback automatique.

use crate::device::GpuDevice;

/// Trait pour les opérations accélérables par GPU
///
/// Ce trait définit une interface pour les algorithmes pouvant être
/// exécutés aussi bien sur CPU que sur GPU. L'implémentation doit
/// garantir que le fallback CPU est toujours disponible.
///
/// # Types associés
///
/// * `Input` - Type des données d'entrée
/// * `Output` - Type des données de sortie
/// * `Error` - Type d'erreur retournée
///
/// # Exemple
///
/// ```rust
/// use pmg_gpu::{GpuAccelerated, GpuDevice};
///
/// struct MaStruct;
///
/// impl GpuAccelerated for MaStruct {
///     type Input = Vec<f64>;
///     type Output = f64;
///     type Error = String;
///
///     fn execute_cpu(&self, input: &Vec<f64>) -> Result<f64, String> {
///         Ok(input.iter().sum())
///     }
///
///     fn execute_gpu(&self, input: &Vec<f64>, _device: &GpuDevice) -> Result<f64, String> {
///         // Fallback sur CPU pour cet exemple
///         self.execute_cpu(input)
///     }
/// }
/// ```
pub trait GpuAccelerated {
    /// Type de données d'entrée
    type Input;

    /// Type de données de sortie
    type Output;

    /// Type d'erreur
    type Error;

    /// Exécute l'opération sur CPU
    ///
    /// # Arguments
    ///
    /// * `input` - Données d'entrée
    ///
    /// # Retour
    ///
    /// Résultat de l'opération ou erreur
    fn execute_cpu(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;

    /// Exécute l'opération sur GPU si disponible
    ///
    /// # Arguments
    ///
    /// * `input` - Données d'entrée
    /// * `device` - Référence vers le device GPU à utiliser
    ///
    /// # Retour
    ///
    /// Résultat de l'opération ou erreur
    ///
    /// # Panique
    ///
    /// Peut paniquer si le device GPU n'est pas valide
    fn execute_gpu(
        &self,
        input: &Self::Input,
        device: &GpuDevice,
    ) -> Result<Self::Output, Self::Error>;

    /// Exécute l'opération avec fallback automatique
    ///
    /// Si un device GPU est fourni, tente l'exécution sur GPU.
    /// Sinon, utilise le fallback CPU. En cas d'échec GPU,
    /// retombe automatiquement sur le CPU.
    ///
    /// # Arguments
    ///
    /// * `input` - Données d'entrée
    /// * `device` - Device GPU optionnel
    ///
    /// # Retour
    ///
    /// Résultat de l'opération ou erreur
    fn execute(
        &self,
        input: &Self::Input,
        device: Option<&GpuDevice>,
    ) -> Result<Self::Output, Self::Error> {
        match device {
            Some(dev) => {
                // Tenter l'exécution GPU, fallback CPU en cas d'échec
                match self.execute_gpu(input, dev) {
                    Ok(result) => Ok(result),
                    Err(_) => self.execute_cpu(input),
                }
            },
            None => self.execute_cpu(input),
        }
    }

    /// Vérifie si GPU est disponible pour cette opération
    ///
    /// Par défaut, retourne `true`. Les implémentations peuvent
    /// surcharger cette méthode pour indiquer si le GPU est supporté.
    fn is_gpu_supported(&self) -> bool {
        true
    }
}
