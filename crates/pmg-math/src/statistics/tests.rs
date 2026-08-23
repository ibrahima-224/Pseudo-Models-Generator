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

//! Tests unitaires pour le module statistiques.

use super::*;
use crate::error::MathError;

#[test]
fn empty_slices_are_rejected() {
    let empty: &[f64] = &[];
    for f in [
        basic_stats::sum,
        basic_stats::mean,
        basic_stats::variance_sample,
        basic_stats::variance_population,
        basic_stats::std_sample,
        basic_stats::std_population,
        basic_stats::skewness,
        basic_stats::kurtosis,
        basic_stats::median,
    ] {
        assert!(matches!(f(empty), Err(MathError::EmptyData(_))));
    }
    assert!(matches!(
        basic_stats::min_max(empty),
        Err(MathError::EmptyData(_))
    ));
    assert!(matches!(
        basic_stats::quantiles(empty, &[0.5]),
        Err(MathError::EmptyData(_))
    ));
    assert!(matches!(
        basic_stats::summary(empty),
        Err(MathError::EmptyData(_))
    ));
}

#[test]
fn known_values_on_constant_data() {
    let data = [5.0, 5.0, 5.0];
    assert_eq!(basic_stats::mean(&data).unwrap(), 5.0);
    assert_eq!(basic_stats::variance_sample(&data).unwrap(), 0.0);
    assert_eq!(basic_stats::variance_population(&data).unwrap(), 0.0);
    assert_eq!(basic_stats::skewness(&data).unwrap(), 0.0);
    assert_eq!(basic_stats::kurtosis(&data).unwrap(), 0.0);
    assert_eq!(basic_stats::min_max(&data).unwrap(), (5.0, 5.0));
}

#[test]
fn known_values_small_sequence() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(basic_stats::mean(&data).unwrap(), 3.0);
    assert_eq!(basic_stats::sum(&data).unwrap(), 15.0);
    assert!((basic_stats::variance_population(&data).unwrap() - 2.0).abs() < 1e-12);
    assert!((basic_stats::variance_sample(&data).unwrap() - 2.5).abs() < 1e-12);
    assert!((basic_stats::std_population(&data).unwrap() - 2.0f64.sqrt()).abs() < 1e-12);
    assert_eq!(basic_stats::min_max(&data).unwrap(), (1.0, 5.0));
    assert!(basic_stats::skewness(&data).unwrap().abs() < 1e-12);
    assert!(basic_stats::kurtosis(&data).unwrap() < 0.0);
}

#[test]
fn variance_sample_matches_formula() {
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert!((basic_stats::variance_population(&data).unwrap() - 4.0).abs() < 1e-12);
    assert!((basic_stats::variance_sample(&data).unwrap() - 32.0 / 7.0).abs() < 1e-12);
}

#[test]
fn skewness_sign_is_correct() {
    let right = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 100.0];
    assert!(basic_stats::skewness(&right).unwrap() > 0.0);
    let left: Vec<f64> = right.iter().map(|v| 101.0 - v).collect();
    assert!(basic_stats::skewness(&left).unwrap() < 0.0);
}

#[test]
fn quantiles_known_values() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let q = basic_stats::quantiles(&data, &[0.0, 0.25, 0.5, 0.75, 1.0]).unwrap();
    assert_eq!(q[0], 1.0);
    assert_eq!(q[1], 2.0);
    assert_eq!(q[2], 3.0);
    assert_eq!(q[3], 4.0);
    assert_eq!(q[4], 5.0);
    let two = [0.0, 10.0];
    assert!((basic_stats::quantiles(&two, &[0.1]).unwrap()[0] - 1.0).abs() < 1e-12);
    assert_eq!(basic_stats::median(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 2.5);
}

#[test]
fn quantile_out_of_range_rejected() {
    assert!(matches!(
        basic_stats::quantiles(&[1.0, 2.0], &[1.5]),
        Err(MathError::InvalidParameter(_))
    ));
}

#[test]
fn quantiles_do_not_mutate_input() {
    let data = [3.0, 1.0, 2.0];
    let _ = basic_stats::quantiles(&data, &[0.5]).unwrap();
    assert_eq!(data, [3.0, 1.0, 2.0]);
}

#[test]
fn summary_matches_individual_functions() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let s = basic_stats::summary(&data).unwrap();
    assert_eq!(s.count, 7);
    assert_eq!(s.mean, basic_stats::mean(&data).unwrap());
    assert_eq!(
        s.variance_sample,
        basic_stats::variance_sample(&data).unwrap()
    );
    assert_eq!(
        s.variance_population,
        basic_stats::variance_population(&data).unwrap()
    );
    assert_eq!(s.std_sample, basic_stats::std_sample(&data).unwrap());
    assert_eq!(s.min, basic_stats::min_max(&data).unwrap().0);
    assert_eq!(s.max, basic_stats::min_max(&data).unwrap().1);
    assert_eq!(s.skewness, basic_stats::skewness(&data).unwrap());
    assert_eq!(s.kurtosis, basic_stats::kurtosis(&data).unwrap());
}

#[test]
fn norm_l1_known_values() {
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_eq!(norms::norm_l1(&data).unwrap(), 8.0);
}

#[test]
fn norm_l2_known_values() {
    let data = [3.0, 4.0];
    assert_eq!(norms::norm_l2(&data).unwrap(), 5.0);
}

#[test]
fn norm_infinity_known_values() {
    let data = [-5.0, 3.0, 7.0];
    assert_eq!(norms::norm_infinity(&data).unwrap(), 7.0);
}

#[test]
fn tail_statistics_known_values() {
    let data = [0.0; 1000];
    let mut data = data.to_vec();
    data[0] = 10.0; // outlier
    let (q99, q999, q9999, ratio) = norms::tail_statistics(&data).unwrap();
    assert!(q99 >= 0.0);
    assert!(q999 >= 0.0);
    assert!(q9999 >= 0.0);
    assert!(ratio > 0.0);
}
