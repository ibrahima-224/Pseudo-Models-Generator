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

//! Module des commandes CLI.
//!
//! Ce module contient toutes les commandes disponibles dans l'interface
//! utilisateur en ligne de commande de PMG.

pub mod compare;
pub mod espec;
pub mod generate;
pub mod generate_blueprint;
pub mod generate_distributed;
pub mod generate_helpers;
pub mod help;
pub mod validate;
pub mod version;

pub use compare::CompareArgs;
pub use espec::EspecArgs;
pub use generate::GenerateArgs;
pub use help::HelpArgs;
pub use validate::ValidateArgs;
pub use version::VersionArgs;
