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

//! Sous-modules pour l'estimation des paramètres et les CDF des distributions.

pub mod log_normal;
pub mod pareto;
pub mod student_t;
pub mod weibull;

pub use log_normal::estimate_lognormal_params;
pub use pareto::estimate_pareto_params;
pub use student_t::estimate_student_t_params;
pub use weibull::estimate_weibull_params;
