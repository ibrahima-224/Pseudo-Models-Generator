# pmg-meta — Manifeste PMG, rapports et provenance

Crate responsible pour la sérialisation/désérialisation des artefacts de sortie du générateur PMG.

## Description

`pmg-meta` fournit les structures de données pour :
- Le manifeste canonique `pmg_metadata.json`
- Les statistiques de génération `pmg/statistics.json`
- Les informations de provenance `pmg/provenance.json`

## Modules

- [`manifest`] : Structure `PmgMetadata` pour le manifeste canonique
- [`statistics`] : Métriques agrégées de génération
- [`provenance`] : Traçabilité des sources et métadonnées

## Utilisation

```rust
use pmg_meta::{PmgMetadata, PmgStatistics, ProvenanceInfo};

// Création d'un manifeste
let metadata = PmgMetadata::new_default();
assert!(metadata.validate().is_ok());

// Sérialisation JSON
let json = metadata.to_json().unwrap();
let deserialized = PmgMetadata::from_json(&json).unwrap();

// Création de statistiques
let stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);

// Création de provenance
let provenance = ProvenanceInfo::new("gen-123", 42, "full");
```

## Validation

Le manifeste est validé selon les règles canoniques :
- Format et version du schéma
- Champs obligatoires non vides
- Cohérence des tailles (target <= estimated <= actual)
- Format du hash et du timestamp

## Dépendances

- `pmg-core`
- `serde` / `serde_json`
- `chrono`
- `thiserror`
- `sha2`
- `num_cpus`

## Tests

```bash
cargo test -p pmg-meta
```

## Licence

Voir le fichier LICENSE à la racine du projet.
