//! Système de progression en temps réel pour la génération de modèles
//!
//! Ce module fournit un suivi détaillé de la progression de la génération
//! incluant barre de progression, statistiques et ETA.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// État de la progression
#[derive(Debug, Clone)]
pub struct ProgressState {
    /// Nombre total d'éléments à traiter
    pub total: usize,
    /// Nombre d'éléments traités
    pub current: usize,
    /// Nom de l'élément en cours
    pub current_item: String,
    /// Temps de début
    pub start_time: Instant,
    /// Dernière mise à jour
    pub last_update: Instant,
    /// Statistiques supplémentaires
    pub stats: ProgressStats,
}

/// Statistiques de progression
#[derive(Debug, Clone, Default)]
pub struct ProgressStats {
    /// Nombre de tenseurs générés
    pub tensors_generated: usize,
    /// Taille totale générée en octets
    pub bytes_generated: u64,
    /// Vitesse moyenne en octets/seconde
    pub avg_speed: f64,
    /// Vitesse instantanée
    pub instant_speed: f64,
    /// Mémoire utilisée (si disponible)
    pub memory_usage: Option<u64>,
    /// ETA en secondes
    pub eta_seconds: Option<f64>,
}

/// Gestionnaire de progression
#[derive(Clone)]
pub struct ProgressTracker {
    /// État actuel
    state: Arc<Mutex<ProgressState>>,
    /// Mode d'affichage
    display_mode: DisplayMode,
    /// Dernière progression affichée
    last_displayed: Arc<Mutex<Option<Instant>>>,
}

/// Mode d'affichage de la progression
#[derive(Debug, Clone, Copy)]
pub enum DisplayMode {
    /// Barre de progression classique
    Bar,
    /// Mode verbeux avec détails
    Verbose,
    /// Mode silencieux (pas de sortie)
    Quiet,
    /// Format JSON pour l'intégration
    Json,
}

impl ProgressTracker {
    /// Crée un nouveau suiveur de progression
    pub fn new(total: usize, display_mode: DisplayMode) -> Self {
        let now = Instant::now();
        Self {
            state: Arc::new(Mutex::new(ProgressState {
                total,
                current: 0,
                current_item: String::new(),
                start_time: now,
                last_update: now,
                stats: ProgressStats::default(),
            })),
            display_mode,
            last_displayed: Arc::new(Mutex::new(None)),
        }
    }

    /// Met à jour la progression
    pub fn update(&self, current: usize, item_name: &str) {
        let mut state = self.state.lock().unwrap();
        state.current = current;
        state.current_item = item_name.to_string();
        state.last_update = Instant::now();

        // Calculer les statistiques
        let elapsed = state
            .last_update
            .duration_since(state.start_time)
            .as_secs_f64();
        if elapsed > 0.0 && current > 0 {
            let total_bytes = state.stats.bytes_generated;
            state.stats.avg_speed = total_bytes as f64 / elapsed;

            // Estimer l'ETA
            if state.stats.avg_speed > 0.0 {
                let remaining_items = state.total - current;
                let avg_bytes_per_item = total_bytes as f64 / current as f64;
                let remaining_bytes = avg_bytes_per_item * remaining_items as f64;
                state.stats.eta_seconds = Some(remaining_bytes / state.stats.avg_speed);
            }
        }

        // Afficher la progression
        self.display_progress(&state);
    }

    /// Met à jour les statistiques
    pub fn update_stats(&self, bytes: u64) {
        let mut state = self.state.lock().unwrap();
        state.stats.bytes_generated += bytes;
        state.stats.tensors_generated += 1;
    }

    /// Affiche la progression
    fn display_progress(&self, state: &ProgressState) {
        match self.display_mode {
            DisplayMode::Bar => self.display_bar(state),
            DisplayMode::Verbose => self.display_verbose(state),
            DisplayMode::Quiet => {},
            DisplayMode::Json => self.display_json(state),
        }
    }

    /// Affiche une barre de progression
    fn display_bar(&self, state: &ProgressState) {
        let mut last = self.last_displayed.lock().unwrap();
        let now = Instant::now();

        // Limiter la fréquence d'affichage (max 10 fois par seconde)
        if let Some(last_time) = *last {
            if now.duration_since(last_time) < Duration::from_millis(100) {
                return;
            }
        }
        *last = Some(now);

        let progress = state.current as f64 / state.total as f64;
        let bar_width = 40;
        let filled = (progress * bar_width as f64) as usize;
        let empty = bar_width - filled;

        let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));
        let percent = progress * 100.0;

        let eta = state
            .stats
            .eta_seconds
            .map(|e| format!(" ETA: {:.0}s", e))
            .unwrap_or_default();

        eprint!(
            "\r{} {:.1}% {}/{}{}",
            bar, percent, state.current, state.total, eta
        );

        if state.current == state.total {
            eprintln!();
        }
    }

    /// Affiche des détails verbeux
    fn display_verbose(&self, state: &ProgressState) {
        let speed = state.stats.avg_speed;

        eprintln!(
            "[{}/{}] {} | Vitesse: {:.2} MB/s | Mémoire: {} | ETA: {}",
            state.current,
            state.total,
            state.current_item,
            speed / 1024.0 / 1024.0,
            state
                .stats
                .memory_usage
                .map(|m| format!("{} MB", m / 1024 / 1024))
                .unwrap_or_else(|| "N/A".to_string()),
            state
                .stats
                .eta_seconds
                .map(|e| format!("{:.1}s", e))
                .unwrap_or_else(|| "N/A".to_string()),
        );
    }

    /// Affiche au format JSON
    fn display_json(&self, state: &ProgressState) {
        let json = serde_json::json!({
            "current": state.current,
            "total": state.total,
            "percent": (state.current as f64 / state.total as f64 * 100.0),
            "item": state.current_item,
            "stats": {
                "tensors": state.stats.tensors_generated,
                "bytes": state.stats.bytes_generated,
                "speed_mbps": state.stats.avg_speed / 1024.0 / 1024.0,
                "eta_seconds": state.stats.eta_seconds,
            }
        });
        eprintln!("{}", json);
    }

    /// Retourne un callback compatible avec les pipelines
    pub fn callback(&self) -> impl Fn(usize, usize, &str) + Send + Sync + 'static {
        let state = Arc::clone(&self.state);
        let display_mode = self.display_mode;
        let last_displayed = Arc::clone(&self.last_displayed);

        move |current, _total, name| {
            // Mettre à jour l'état
            let mut state_lock = state.lock().unwrap();
            state_lock.current = current;
            state_lock.current_item = name.to_string();
            state_lock.last_update = Instant::now();

            // Calculer les statistiques
            let elapsed = state_lock
                .last_update
                .duration_since(state_lock.start_time)
                .as_secs_f64();
            if elapsed > 0.0 && current > 0 {
                let total_bytes = state_lock.stats.bytes_generated;
                state_lock.stats.avg_speed = total_bytes as f64 / elapsed;

                // Estimer l'ETA
                if state_lock.stats.avg_speed > 0.0 {
                    let remaining_items = state_lock.total - current;
                    let avg_bytes_per_item = total_bytes as f64 / current as f64;
                    let remaining_bytes = avg_bytes_per_item * remaining_items as f64;
                    state_lock.stats.eta_seconds =
                        Some(remaining_bytes / state_lock.stats.avg_speed);
                }
            }

            // Afficher la progression selon le mode
            match display_mode {
                DisplayMode::Bar => {
                    let mut last = last_displayed.lock().unwrap();
                    let now = Instant::now();

                    // Limiter la fréquence d'affichage (max 10 fois par seconde)
                    if let Some(last_time) = *last {
                        if now.duration_since(last_time) < Duration::from_millis(100) {
                            return;
                        }
                    }
                    *last = Some(now);

                    let progress = state_lock.current as f64 / state_lock.total as f64;
                    let bar_width = 40;
                    let filled = (progress * bar_width as f64) as usize;
                    let empty = bar_width - filled;

                    let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));
                    let percent = progress * 100.0;

                    let eta = state_lock
                        .stats
                        .eta_seconds
                        .map(|e| format!(" ETA: {:.0}s", e))
                        .unwrap_or_default();

                    eprint!(
                        "\r{} {:.1}% {}/{}{}",
                        bar, percent, state_lock.current, state_lock.total, eta
                    );

                    if state_lock.current == state_lock.total {
                        eprintln!();
                    }
                },
                DisplayMode::Verbose => {
                    let speed = state_lock.stats.avg_speed;

                    eprintln!(
                        "[{}/{}] {} | Vitesse: {:.2} MB/s | Mémoire: {} | ETA: {}",
                        state_lock.current,
                        state_lock.total,
                        state_lock.current_item,
                        speed / 1024.0 / 1024.0,
                        state_lock
                            .stats
                            .memory_usage
                            .map(|m| format!("{} MB", m / 1024 / 1024))
                            .unwrap_or_else(|| "N/A".to_string()),
                        state_lock
                            .stats
                            .eta_seconds
                            .map(|e| format!("{:.1}s", e))
                            .unwrap_or_else(|| "N/A".to_string()),
                    );
                },
                DisplayMode::Quiet => {},
                DisplayMode::Json => {
                    let json = serde_json::json!({
                        "current": state_lock.current,
                        "total": state_lock.total,
                        "percent": (state_lock.current as f64 / state_lock.total as f64 * 100.0),
                        "item": state_lock.current_item,
                        "stats": {
                            "tensors": state_lock.stats.tensors_generated,
                            "bytes": state_lock.stats.bytes_generated,
                            "speed_mbps": state_lock.stats.avg_speed / 1024.0 / 1024.0,
                            "eta_seconds": state_lock.stats.eta_seconds,
                        }
                    });
                    eprintln!("{}", json);
                },
            }
        }
    }

    /// Finalise l'affichage
    pub fn finish(&self) {
        let state = self.state.lock().unwrap();
        let elapsed = state.last_update.duration_since(state.start_time);

        match self.display_mode {
            DisplayMode::Bar => {
                eprintln!("\n✅ Terminé en {:.2}s", elapsed.as_secs_f64());
            },
            DisplayMode::Verbose => {
                eprintln!("\n✅ Génération terminée:");
                eprintln!("   - Tenseurs: {}", state.stats.tensors_generated);
                eprintln!(
                    "   - Taille: {:.2} MB",
                    state.stats.bytes_generated as f64 / 1024.0 / 1024.0
                );
                eprintln!("   - Durée: {:.2}s", elapsed.as_secs_f64());
                eprintln!(
                    "   - Vitesse moyenne: {:.2} MB/s",
                    state.stats.avg_speed / 1024.0 / 1024.0
                );
            },
            DisplayMode::Json => {
                let json = serde_json::json!({
                    "status": "completed",
                    "duration_seconds": elapsed.as_secs_f64(),
                    "stats": {
                        "tensors": state.stats.tensors_generated,
                        "bytes": state.stats.bytes_generated,
                    }
                });
                eprintln!("{}", json);
            },
            _ => {},
        }
    }
}

/// Crée un suiveur de progression à partir des arguments CLI
/// Note: Cette fonction sera adaptée pour correspondre à la structure GenerateArgs
/// du module CLI. Pour l'instant, elle prend des paramètres génériques.
pub fn create_progress_tracker(
    total: usize,
    quiet: bool,
    json_output: bool,
    verbose: bool,
) -> ProgressTracker {
    let mode = if quiet {
        DisplayMode::Quiet
    } else if json_output {
        DisplayMode::Json
    } else if verbose {
        DisplayMode::Verbose
    } else {
        DisplayMode::Bar
    };

    ProgressTracker::new(total, mode)
}

/// Formatage de la durée
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}

/// Formatage de la taille
pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.2} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(100, DisplayMode::Bar);
        let state = tracker.state.lock().unwrap();
        assert_eq!(state.total, 100);
        assert_eq!(state.current, 0);
    }

    #[test]
    fn test_progress_update() {
        let tracker = ProgressTracker::new(100, DisplayMode::Quiet);
        tracker.update(50, "test_tensor");

        let state = tracker.state.lock().unwrap();
        assert_eq!(state.current, 50);
        assert_eq!(state.current_item, "test_tensor");
    }

    #[test]
    fn test_stats_update() {
        let tracker = ProgressTracker::new(100, DisplayMode::Quiet);
        tracker.update_stats(1024);
        tracker.update_stats(2048);

        let state = tracker.state.lock().unwrap();
        assert_eq!(state.stats.bytes_generated, 3072);
        assert_eq!(state.stats.tensors_generated, 2);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1048576), "1.00 MB");
        assert_eq!(format_size(1073741824), "1.00 GB");
    }
}
