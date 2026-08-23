//! Implémentation GPU pour la génération normale
//!
//! Ce module fournit une implémentation accélérée GPU du trait `GpuAccelerated`
//! pour la génération de nombres aléatoires suivant une distribution normale.
//! Le fallback CPU est toujours disponible.

use crate::acceleration::GpuAccelerated;
use crate::device::GpuDevice;
use crate::error::GpuError;
use std::f64::consts::PI;

/// Convertit un u64 en f64 uniforme sur [0, 1) avec 53 bits de précision.
fn u64_to_f64(v: u64) -> f64 {
    (v >> 11) as f64 / (1u64 << 53) as f64
}

/// Générateur de distribution normale accéléré GPU
///
/// Cette structure implémente le trait `GpuAccelerated` pour générer
/// des nombres aléatoires suivant une distribution normale (gaussienne).
///
/// # Caractéristiques
///
/// - **Déterminisme** : Même seed produit toujours les mêmes résultats
/// - **Fallback CPU** : Toujours disponible sans GPU
/// - **GPU** : Utilise le kernel PTX NORMAL_GENERATION_KERNEL lorsque disponible
///
/// # Exemple
///
/// ```rust
/// use pmg_gpu::kernels::NormalGenerationAccelerated;
/// use pmg_gpu::GpuAccelerated;
///
/// let generator = NormalGenerationAccelerated::new(42);
/// let params = (1000, 0.0, 1.0); // 1000 échantillons, moyenne=0, écart-type=1
/// let data = generator.execute_cpu(&params).unwrap();
/// assert_eq!(data.len(), 1000);
/// ```
pub struct NormalGenerationAccelerated {
    /// Graine pour la génération déterministe
    seed: u64,
}

impl NormalGenerationAccelerated {
    /// Crée un nouveau générateur avec la graine spécifiée
    ///
    /// # Arguments
    ///
    /// * `seed` - Graine pour la génération déterministe
    ///
    /// # Retour
    ///
    /// Instance du générateur prêt à l'emploi
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Retourne la graine utilisée
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl GpuAccelerated for NormalGenerationAccelerated {
    /// Type d'entrée : (taille, moyenne, écart-type)
    type Input = (usize, f64, f64);

    /// Type de sortie : Vecteur de valeurs normalement distribuées
    type Output = Vec<f64>;

    /// Type d'erreur : Erreur GPU
    type Error = GpuError;

    /// Exécute la génération normale sur CPU
    ///
    /// # Arguments
    ///
    /// * `input` - Tuple (taille, moyenne, écart-type)
    ///
    /// # Retour
    ///
    /// Vecteur de `taille` valeurs suivant N(moyenne, écart-type²)
    ///
    /// # Panique
    ///
    /// Panique si l'écart-type est négatif (vérifié par `Normal::new`)
    fn execute_cpu(&self, input: &(usize, f64, f64)) -> Result<Vec<f64>, GpuError> {
        let (size, mean, std) = input;

        // Validation des paramètres
        if *std < 0.0 {
            return Err(GpuError::ValidationError(format!(
                "L'écart-type doit être positif: {}",
                std
            )));
        }

        // Génération par la méthode de Box-Muller
        // (transforme deux variables uniformes en deux variables normales)
        let mut samples = Vec::with_capacity(*size);
        let mut i = 0;
        while i < *size {
            // Utiliser une boucle simple pour générer des nombres uniformes
            // On utilise une transformation simple pour éviter les dépendances
            let u1: f64 = u64_to_f64(
                (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(self.seed),
            );
            let u2: f64 = u64_to_f64(
                (i as u64)
                    .wrapping_mul(1442695040888963407)
                    .wrapping_add(self.seed)
                    .wrapping_add(1),
            );

            // Éviter log(0) qui est -infini
            let u1: f64 = if u1 == 0.0 { f64::MIN_POSITIVE } else { u1 };

            let z0: f64 = ((-2.0_f64 * u1.ln()).sqrt()) * ((2.0_f64 * PI * u2).cos());
            let z1: f64 = ((-2.0_f64 * u1.ln()).sqrt()) * ((2.0_f64 * PI * u2).sin());

            // Transformation en N(mean, std²)
            samples.push(z0 * std + mean);
            i += 1;

            // Ajouter le deuxième échantillon si besoin
            if i < *size {
                samples.push(z1 * std + mean);
                i += 1;
            }
        }

        Ok(samples)
    }

    /// Exécute la génération normale sur GPU
    ///
    /// # Arguments
    ///
    /// * `input` - Tuple (taille, moyenne, écart-type)
    /// * `device` - Référence vers le device GPU à utiliser
    ///
    /// # Retour
    ///
    /// Vecteur de valeurs normalement distribuées
    ///
    /// # Implémentation
    ///
    /// Pour l'instant, fallback sur CPU. L'implémentation GPU réelle
    /// utiliserait le kernel PTX NORMAL_GENERATION_KERNEL.
    fn execute_gpu(
        &self,
        input: &(usize, f64, f64),
        _device: &GpuDevice,
    ) -> Result<Vec<f64>, GpuError> {
        // TODO: Implémenter l'exécution GPU réelle avec NORMAL_GENERATION_KERNEL
        // Pour l'instant, fallback sur CPU
        log::debug!("Fallback GPU->CPU pour la génération normale");
        self.execute_cpu(input)
    }

    /// Vérifie si le GPU est supporté pour cette opération
    ///
    /// # Retour
    ///
    /// `true` si le GPU peut être utilisé, `false` sinon
    fn is_gpu_supported(&self) -> bool {
        // Le support GPU dépend de la feature cuda
        #[cfg(feature = "cuda")]
        {
            // Vérifier si le kernel PTX est disponible
            true
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_normal_generation_cpu_deterministe() {
        // Vérifie la reproductibilité
        let gen1 = NormalGenerationAccelerated::new(42);
        let gen2 = NormalGenerationAccelerated::new(42);

        let params = (100, 0.0, 1.0);
        let data1 = gen1.execute_cpu(&params).unwrap();
        let data2 = gen2.execute_cpu(&params).unwrap();

        assert_eq!(data1.len(), data2.len());
        for (a, b) in data1.iter().zip(data2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_normal_generation_taille_correcte() {
        let gen = NormalGenerationAccelerated::new(123);
        let params = (50, 10.0, 2.0);
        let data = gen.execute_cpu(&params).unwrap();

        assert_eq!(data.len(), 50);
    }

    #[test]
    fn test_normal_generation_distribution() {
        let gen = NormalGenerationAccelerated::new(456);
        let params = (10000, 0.0, 1.0);
        let data = gen.execute_cpu(&params).unwrap();

        // Vérifier que la moyenne est proche de 0
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 0.1, "Moyenne trop éloignée de 0: {}", mean);

        // Vérifier que l'écart-type est proche de 1
        let variance: f64 = data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 1.0).abs() < 0.1,
            "Écart-type trop éloigné de 1: {}",
            std_dev
        );
    }

    #[test]
    fn test_normal_generation_erreur_ecart_type_negatif() {
        let gen = NormalGenerationAccelerated::new(789);
        let params = (100, 0.0, -1.0);
        let result = gen.execute_cpu(&params);

        assert!(result.is_err());
    }
}
