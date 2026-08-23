//! # Surveillance de la mémoire
//!
//! Ce module fournit un moniteur de mémoire pour suivre l'utilisation
//! de la mémoire pendant la génération de modèles en mode streaming.

use std::fmt;

/// Métriques de surveillance mémoire.
///
/// Ces métriques permettent de suivre l'utilisation de la mémoire
/// et de détecter les dépassements de limites.
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// Mémoire maximale autorisée (octets).
    pub max_allowed: u64,
    /// Mémoire maximale utilisée (octets).
    pub peak_usage: u64,
    /// Mémoire actuelle (octets).
    pub current_usage: u64,
    /// Nombre total d'allocations.
    pub allocation_count: usize,
    /// Nombre de réutilisations de buffers.
    pub buffer_reuses: usize,
}

impl MemoryMetrics {
    /// Crée des métriques vides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Crée des métriques avec une limite maximale.
    pub fn with_limit(max_allowed: u64) -> Self {
        Self {
            max_allowed,
            ..Default::default()
        }
    }

    /// Vérifie si la mémoire est dans les limites.
    pub fn is_within_limits(&self) -> bool {
        self.current_usage <= self.max_allowed
    }

    /// Retourne le pourcentage d'utilisation.
    pub fn usage_percentage(&self) -> f64 {
        if self.max_allowed == 0 {
            return 0.0;
        }
        (self.current_usage as f64 / self.max_allowed as f64) * 100.0
    }

    /// Met à jour l'utilisation mémoire.
    pub fn update_usage(&mut self, additional: u64) {
        self.current_usage += additional;
        self.allocation_count += 1;

        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    /// Libère de la mémoire.
    pub fn release(&mut self, amount: u64) {
        if self.current_usage >= amount {
            self.current_usage -= amount;
        } else {
            self.current_usage = 0;
        }
    }

    /// Incrémente le compteur de réutilisations de buffers.
    pub fn increment_buffer_reuse(&mut self) {
        self.buffer_reuses += 1;
    }

    /// Réinitialise les métriques.
    pub fn reset(&mut self) {
        self.current_usage = 0;
        self.peak_usage = 0;
        self.allocation_count = 0;
        self.buffer_reuses = 0;
    }

    /// Affiche les métriques en mode verbose.
    pub fn print_summary(&self) {
        eprintln!("📊 Métriques de mémoire:");
        eprintln!(
            "   - Limite maximale: {:.2} Mo",
            self.max_allowed as f64 / 1024.0 / 1024.0
        );
        eprintln!(
            "   - Utilisation actuelle: {:.2} Mo",
            self.current_usage as f64 / 1024.0 / 1024.0
        );
        eprintln!(
            "   - Pic d'utilisation: {:.2} Mo",
            self.peak_usage as f64 / 1024.0 / 1024.0
        );
        eprintln!("   - Pourcentage utilisé: {:.1}%", self.usage_percentage());
        eprintln!("   - Nombre d'allocations: {}", self.allocation_count);
        eprintln!("   - Réutilisations de buffers: {}", self.buffer_reuses);
    }
}

impl fmt::Display for MemoryMetrics {
    /// Affiche les métriques de manière lisible.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryMetrics {{ max: {:.2} Mo, current: {:.2} Mo, peak: {:.2} Mo, usage: {:.1}%, allocations: {}, reuses: {} }}",
            self.max_allowed as f64 / 1024.0 / 1024.0,
            self.current_usage as f64 / 1024.0 / 1024.0,
            self.peak_usage as f64 / 1024.0 / 1024.0,
            self.usage_percentage(),
            self.allocation_count,
            self.buffer_reuses
        )
    }
}

/// Moniteur de mémoire pour la génération de modèles.
///
/// Ce moniteur suit l'utilisation de la mémoire pendant la génération
/// et permet de détecter les dépassements de limites.
pub struct MemoryMonitor {
    /// Métriques de mémoire.
    metrics: MemoryMetrics,
    /// Seuil d'avertissement (défaut : 80%).
    warning_threshold: f64,
    /// Seuil critique (défaut : 95%).
    critical_threshold: f64,
    /// Mode verbose (défaut : false).
    verbose: bool,
}

impl MemoryMonitor {
    /// Crée un nouveau moniteur de mémoire.
    pub fn new(max_memory: u64) -> Self {
        Self {
            metrics: MemoryMetrics::with_limit(max_memory),
            warning_threshold: 80.0,
            critical_threshold: 95.0,
            verbose: false,
        }
    }

    /// Crée un moniteur avec des seuils personnalisés.
    pub fn with_thresholds(
        max_memory: u64,
        warning_threshold: f64,
        critical_threshold: f64,
    ) -> Self {
        Self {
            metrics: MemoryMetrics::with_limit(max_memory),
            warning_threshold,
            critical_threshold,
            verbose: false,
        }
    }

    /// Active le mode verbose.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Vérifie si une allocation peut être effectuée.
    ///
    /// # Paramètres
    /// - `size` : taille de l'allocation en octets
    ///
    /// # Retourne
    /// `true` si l'allocation est possible, `false` sinon.
    pub fn can_allocate(&self, size: u64) -> bool {
        self.metrics.current_usage + size <= self.metrics.max_allowed
    }

    /// Enregistre une allocation mémoire.
    ///
    /// # Paramètres
    /// - `size` : taille de l'allocation en octets
    ///
    /// # Retourne
    /// `true` si l'allocation est dans les limites, `false` sinon.
    pub fn allocate(&mut self, size: u64) -> bool {
        if !self.can_allocate(size) {
            if self.verbose {
                eprintln!(
                    "⚠️  Dépassement de limite mémoire: allocation de {:.2} Mo refusée",
                    size as f64 / 1024.0 / 1024.0
                );
            }
            return false;
        }

        self.metrics.update_usage(size);

        // Vérification des seuils
        let usage = self.metrics.usage_percentage();

        if usage >= self.critical_threshold && self.verbose {
            eprintln!(
                "🔴 CRITIQUE: Utilisation mémoire à {:.1}% ({:.2} Mo / {:.2} Mo)",
                usage,
                self.metrics.current_usage as f64 / 1024.0 / 1024.0,
                self.metrics.max_allowed as f64 / 1024.0 / 1024.0
            );
        } else if usage >= self.warning_threshold && self.verbose {
            eprintln!(
                "🟡 ATTENTION: Utilisation mémoire à {:.1}% ({:.2} Mo / {:.2} Mo)",
                usage,
                self.metrics.current_usage as f64 / 1024.0 / 1024.0,
                self.metrics.max_allowed as f64 / 1024.0 / 1024.0
            );
        }

        true
    }

    /// Libère de la mémoire.
    pub fn release(&mut self, size: u64) {
        self.metrics.release(size);
    }

    /// Retourne les métriques actuelles.
    pub fn metrics(&self) -> &MemoryMetrics {
        &self.metrics
    }

    /// Retourne le pourcentage d'utilisation actuel.
    pub fn usage_percentage(&self) -> f64 {
        self.metrics.usage_percentage()
    }

    /// Vérifie si on est près de la limite (> 80%).
    pub fn is_near_limit(&self) -> bool {
        self.metrics.usage_percentage() >= self.warning_threshold
    }

    /// Vérifie si on est en dépassement.
    pub fn is_over_limit(&self) -> bool {
        !self.metrics.is_within_limits()
    }

    /// Réinitialise le moniteur.
    pub fn reset(&mut self) {
        self.metrics.reset();
    }

    /// Affiche un résumé des métriques.
    pub fn print_summary(&self) {
        self.metrics.print_summary();
    }
}

impl fmt::Display for MemoryMonitor {
    /// Affiche l'état du moniteur de manière lisible.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryMonitor {{ usage: {:.1}%, near_limit: {}, over_limit: {} }}",
            self.usage_percentage(),
            self.is_near_limit(),
            self.is_over_limit()
        )
    }
}

// ============================================================================
// Tests unitaires
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test de création de MemoryMetrics
    #[test]
    fn test_memory_metrics_new() {
        let metrics = MemoryMetrics::new();
        assert_eq!(metrics.max_allowed, 0);
        assert_eq!(metrics.current_usage, 0);
        assert_eq!(metrics.peak_usage, 0);
    }

    /// Test de création avec limite
    #[test]
    fn test_memory_metrics_with_limit() {
        let metrics = MemoryMetrics::with_limit(1024 * 1024 * 1024); // 1 Go
        assert_eq!(metrics.max_allowed, 1024 * 1024 * 1024);
        assert!(metrics.is_within_limits());
    }

    /// Test de mise à jour de l'utilisation
    #[test]
    fn test_memory_metrics_update_usage() {
        let mut metrics = MemoryMetrics::with_limit(1024 * 1024 * 1024);

        metrics.update_usage(100 * 1024 * 1024); // 100 Mo
        assert_eq!(metrics.current_usage, 100 * 1024 * 1024);
        assert_eq!(metrics.peak_usage, 100 * 1024 * 1024);
        assert_eq!(metrics.allocation_count, 1);

        metrics.update_usage(200 * 1024 * 1024); // 200 Mo supplémentaires
        assert_eq!(metrics.current_usage, 300 * 1024 * 1024);
        assert_eq!(metrics.peak_usage, 300 * 1024 * 1024);
        assert_eq!(metrics.allocation_count, 2);
    }

    /// Test de libération de mémoire
    #[test]
    fn test_memory_metrics_release() {
        let mut metrics = MemoryMetrics::with_limit(1024 * 1024 * 1024);

        metrics.update_usage(300 * 1024 * 1024);
        metrics.release(100 * 1024 * 1024);

        assert_eq!(metrics.current_usage, 200 * 1024 * 1024);
    }

    /// Test du pourcentage d'utilisation
    #[test]
    fn test_memory_metrics_usage_percentage() {
        let mut metrics = MemoryMetrics::with_limit(1000);

        metrics.update_usage(500);
        assert!((metrics.usage_percentage() - 50.0).abs() < f64::EPSILON);

        metrics.update_usage(300);
        assert!((metrics.usage_percentage() - 80.0).abs() < f64::EPSILON);
    }

    /// Test de création de MemoryMonitor
    #[test]
    fn test_memory_monitor_new() {
        let monitor = MemoryMonitor::new(500 * 1024 * 1024);

        assert_eq!(monitor.metrics().max_allowed, 500 * 1024 * 1024);
        assert!(!monitor.is_near_limit());
        assert!(!monitor.is_over_limit());
    }

    /// Test d'allocation mémoire
    #[test]
    fn test_memory_monitor_allocate() {
        let mut monitor = MemoryMonitor::new(1000);

        // Allocation réussie
        assert!(monitor.allocate(500));
        assert_eq!(monitor.metrics().current_usage, 500);

        // Allocation qui dépasse la limite
        assert!(!monitor.allocate(600));
        assert_eq!(monitor.metrics().current_usage, 500); // Pas de changement
    }

    /// Test des seuils d'avertissement
    #[test]
    fn test_memory_monitor_thresholds() {
        let mut monitor = MemoryMonitor::new(1000);

        // Allocation à 80% (seuil d'avertissement)
        monitor.allocate(800);
        assert!(monitor.is_near_limit());
        assert!(!monitor.is_over_limit()); // Pas encore en dépassement

        // Vérification que l'allocation de 150 échoue car elle dépasse la limite
        // (800 + 150 = 950, mais la méthode allocate vérifie current_usage + size <= max_allowed)
        // Actually 800 + 150 = 950 <= 1000, so allocation should succeed
        let result = monitor.allocate(150);
        assert!(result); // L'allocation réussit car 950 <= 1000
        assert!(!monitor.is_over_limit()); // 950 <= 1000, donc pas en dépassement

        // Forcer un dépassement via update_usage direct sur les métriques
        // pour tester is_over_limit()
        monitor.metrics.update_usage(100); // 950 + 100 = 1050 > 1000
        assert!(monitor.is_over_limit()); // Maintenant en dépassement
    }

    /// Test de vérification avant allocation
    #[test]
    fn test_memory_monitor_can_allocate() {
        let monitor = MemoryMonitor::new(1000);

        assert!(monitor.can_allocate(500));
        assert!(monitor.can_allocate(1000));
        assert!(!monitor.can_allocate(1001));
    }
}
