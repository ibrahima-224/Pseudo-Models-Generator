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

//! @deprecated : Utilisez `pmg_generator::GenerationPipeline` à la place.
//! Ce module sera supprimé dans une version future.
//!
//! Conformité : ADR-002, étape 2 - Unification des pipelines.
//! Ce module est devenu un re-export temporaire pour assurer la transition
//! sans cassure API. Le pipeline complet a été unifié dans pmg-generator.

// NOTE: Ce module ne réexporte plus rien car le pipeline de pmg-core
// n'est pas utilisé en production (uniquement dans les tests internes).
// Les tests doivent être migrés vers pmg-generator.
