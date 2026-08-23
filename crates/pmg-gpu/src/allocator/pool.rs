//! Allocations pool pour réduire la fragmentation
//!
//! Ce module fournit un allocateur pool qui pré-alloue des blocs de mémoire
//! pour réduire la fragmentation et améliorer les performances des allocations fréquentes.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::device::GpuPointer;
use crate::error::GpuResult;

use super::gpu_allocator::{GpuAllocator, GpuAllocatorBlock};
use super::stats::PoolStats;

/// Allocations pool pour réduire la fragmentation
///
/// Fournit un pool de mémoire pré-allouée pour les allocations fréquentes.
pub struct GpuPoolAllocator {
    /// Allocateur sous-jacent
    allocator: Arc<GpuAllocator>,
    /// Pool de blocs libres organisés par taille
    pools: Mutex<BTreeMap<usize, Vec<GpuPointer>>>,
    /// Taille maximale des blocs dans le pool
    max_pool_size: usize,
    /// Nombre maximum de blocs par taille
    max_blocks_per_size: usize,
}

impl GpuPoolAllocator {
    /// Crée un nouvel allocateur pool
    ///
    /// # Arguments
    ///
    /// * `allocator` - Allocateur sous-jacent
    /// * `max_pool_size` - Taille maximale des blocs gérés par le pool
    /// * `max_blocks_per_size` - Nombre maximum de blocs par taille
    pub fn new(
        allocator: Arc<GpuAllocator>,
        max_pool_size: usize,
        max_blocks_per_size: usize,
    ) -> Self {
        Self {
            allocator,
            pools: Mutex::new(BTreeMap::new()),
            max_pool_size,
            max_blocks_per_size,
        }
    }

    /// Alloue un bloc depuis le pool ou l'allocateur sous-jacent
    ///
    /// # Arguments
    ///
    /// * `size` - Taille en octets à allouer
    ///
    /// # Retour
    ///
    /// Un bloc de mémoire alloué encapsulé dans un `GpuAllocatorBlock`.
    pub fn allocate(&self, size: usize) -> GpuResult<GpuAllocatorBlock> {
        // Arrondir à la taille supérieure (multiples de 256 octets)
        let aligned_size = (size + 255) & !255;

        // Chercher dans le pool
        if let Ok(mut pools) = self.pools.lock() {
            if let Some(blocks) = pools.get_mut(&aligned_size) {
                if let Some(pointer) = blocks.pop() {
                    log::debug!("Allocation depuis le pool: {} octets", aligned_size);

                    return Ok(GpuAllocatorBlock {
                        pointer,
                        allocator: Arc::clone(&self.allocator),
                    });
                }
            }
        }

        // Allouer via l'allocateur sous-jacent
        Arc::clone(&self.allocator).allocate(aligned_size, Some("pool"))
    }

    /// Retourne un bloc au pool
    ///
    /// # Arguments
    ///
    /// * `block` - Bloc à retourner au pool
    ///
    /// # Retour
    ///
    /// `true` si le bloc a été ajouté au pool, `false` sinon
    pub fn release(&self, block: GpuAllocatorBlock) -> bool {
        let size = block.pointer.size;

        // Vérifier si le bloc peut être mis en pool
        if size > self.max_pool_size {
            // Libérer directement
            drop(block);
            return false;
        }

        // Ajouter au pool
        if let Ok(mut pools) = self.pools.lock() {
            let blocks = pools.entry(size).or_insert_with(Vec::new);

            if blocks.len() < self.max_blocks_per_size {
                // Conserver le pointeur sans le libérer
                // Utiliser ManuallyDrop pour éviter le drop automatique
                // SAFETY : block est validé non-null et aligné avant utilisation
                // Le pointeur est extrait sans libération pour stockage dans le pool
                let block = std::mem::ManuallyDrop::new(block);

                // Vérifications de validité du pointeur avant extraction
                // Ces vérifications ne sont possibles qu'avec la feature gpu
                #[cfg(feature = "gpu")]
                {
                    let raw_ptr = block.pointer.raw_ptr();
                    // Vérifier que le pointeur n'est pas null (0 pour CUDA)
                    debug_assert!(raw_ptr != 0, "Le pointeur GPU ne doit pas être null");
                    // Vérifier l'alignement (les allocations CUDA sont alignées sur 256 octets)
                    debug_assert!(
                        raw_ptr % 256 == 0,
                        "Le pointeur GPU doit être aligné sur 256 octets"
                    );
                }
                // Vérification de la taille (toujours possible)
                debug_assert!(
                    block.pointer.size > 0,
                    "La taille du bloc doit être positive"
                );

                // SAFETY : Cette opération unsafe est nécessaire pour extraire le pointeur sans
                // déclencher la libération automatique de la mémoire GPU.
                //
                // Pourquoi ManuallyDrop :
                // - Le bloc `block` contient un pointeur GPU valide que nous voulons stocker dans le pool
                //   pour une réutilisation future. Si nous laissions le drop automatique s'exécuter,
                //   la mémoire GPU serait libérée, rendant le pointeur invalide.
                // - ManuallyDrop empêche l'exécution du destructeur de `block` lorsqu'il sort de portée.
                //
                // Pourquoi std::ptr::read :
                // - Nous avons besoin de copier le contenu du pointeur (l'adresse GPU et la taille)
                //   sans déplacer le bloc original (qui est toujours géré par ManuallyDrop).
                // - ptr::read effectue une copie bit-à-bit du pointeur, créant une nouvelle instance
                //   de GpuPointer que nous pouvons stocker dans le pool.
                //
                // Invariants de sécurité vérifiés avant cette opération :
                // 1. block.pointer est validé non-null (vérifié par debug_assert précédemment)
                // 2. block.pointer est correctement aligné (vérifié par debug_assert précédemment)
                // 3. La taille du pointeur est positive (vérifié par debug_assert précédemment)
                // 4. Le type GpuPointer est 'Pod' (plain old data) et peut être copié de manière sûre
                //
                // Conséquences :
                // - Après cette opération, `block` contient toujours le pointeur original, mais
                //   celui-ci ne sera pas libéré car il est wrappé dans ManuallyDrop.
                // - `pointer` est une copie indépendante du pointeur que nous stockons dans le pool.
                // - Lorsque `block` sortira de portée, seul le wrapper ManuallyDrop sera détruit,
                //   pas le pointeur GPU sous-jacent.
                let pointer = unsafe { std::ptr::read(&block.pointer) };

                // Vérification post-extraction
                debug_assert_eq!(
                    pointer.size, size,
                    "La taille extraite doit correspondre à la taille demandée"
                );

                blocks.push(pointer);
                log::debug!("Retour au pool: {} octets", size);
                return true;
            }
        }

        // Pool plein, libérer directement
        drop(block);
        false
    }

    /// Nettoie le pool en libérant les blocs inutilisés
    pub fn cleanup(&self) {
        if let Ok(mut pools) = self.pools.lock() {
            let total_blocks: usize = pools.values().map(|v| v.len()).sum();
            pools.clear();

            if total_blocks > 0 {
                log::info!("Nettoyage pool: {} blocs libérés", total_blocks);
            }
        }
    }

    /// Retourne des statistiques sur le pool
    pub fn pool_stats(&self) -> PoolStats {
        let mut stats = PoolStats::default();

        if let Ok(pools) = self.pools.lock() {
            for (size, blocks) in pools.iter() {
                stats.total_blocks += blocks.len();
                stats.total_memory += size * blocks.len();

                if let Some(&min_size) = pools.keys().next() {
                    stats.min_block_size = min_size;
                }
                if let Some(&max_size) = pools.keys().next_back() {
                    stats.max_block_size = max_size;
                }
            }
        }

        stats
    }
}
