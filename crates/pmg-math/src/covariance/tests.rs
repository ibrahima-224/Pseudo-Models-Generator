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

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::covariance::*;
    use crate::error::MathError;
    use crate::rng::DeterministicRng;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([23u8; 32])
    }

    #[test]
    fn diagonal_case() {
        let c = Covariance::diagonal(&[1.0, 4.0, 9.0]).unwrap();
        assert_eq!(c.cholesky().dim, 3);
        assert!(c.reconstruction_error() < 1e-12);
        // L = diag(1, 2, 3).
        let l = &c.cholesky().l;
        assert_eq!(l[0], 1.0);
        assert_eq!(l[3 + 1], 2.0);
        assert_eq!(l[2 * 3 + 2], 3.0);
    }

    #[test]
    fn diagonal_rejects_non_positive_variances() {
        assert!(Covariance::diagonal(&[0.0]).is_err());
        assert!(Covariance::diagonal(&[-1.0, 2.0]).is_err());
        assert!(Covariance::diagonal(&[]).is_err());
    }

    #[test]
    fn two_by_two_cholesky() {
        // Σ = [[2, 0.5], [0.5, 1]] → L = [[√2, 0], [0.5/√2, √(1−0.125)]].
        let c = Covariance::new(vec![2.0, 0.5, 0.5, 1.0], 2).unwrap();
        assert!(c.reconstruction_error() < 1e-12);
        let l = &c.cholesky().l;
        assert!((l[0] - 2.0f64.sqrt()).abs() < 1e-12);
        assert!((l[2] - 0.5 / 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn one_by_one() {
        let c = Covariance::new(vec![4.0], 1).unwrap();
        assert_eq!(c.cholesky().l[0], 2.0);
        assert!(c.reconstruction_error() < 1e-12);
    }

    #[test]
    fn non_psd_is_rejected_explicitly() {
        // Matrice avec une valeur propre négative : [[1, 2], [2, 1]].
        let bad = Covariance::new(vec![1.0, 2.0, 2.0, 1.0], 2);
        assert!(matches!(bad, Err(MathError::NotPsd(_))));
        // Asymétrie.
        let asym = Covariance::new(vec![1.0, 2.0, 3.0, 1.0], 2);
        assert!(matches!(asym, Err(MathError::NotPsd(_))));
        // Mauvais format.
        assert!(matches!(
            Covariance::new(vec![1.0, 2.0, 3.0], 2),
            Err(MathError::InvalidParameter(_))
        ));
        assert!(matches!(
            Covariance::new(vec![], 0),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn covariance_matches_empirical() {
        // Σ = [[2, 0.5], [0.5, 1]], n = 100_000 : covariance empirique ≈ Σ.
        let cov = Covariance::new(vec![2.0, 0.5, 0.5, 1.0], 2).unwrap();
        let mut rng = rng();
        let n = 100_000;
        let samples = sample_correlated(&mut rng, &[0.0, 0.0], cov.cholesky(), n).unwrap();
        let dim = 2;
        let nf = n as f64;
        // Moyennes empiriques par composante.
        let mut means = vec![0.0; dim];
        for i in 0..dim {
            means[i] = samples[i..].iter().step_by(dim).sum::<f64>() / nf;
        }
        // Calcul direct de la matrice de covariance empirique (diviseur n−1).
        let mut accum = vec![0.0; dim * dim];
        for row in 0..n {
            let base = row * dim;
            for a in 0..dim {
                for b in 0..dim {
                    accum[a * dim + b] +=
                        (samples[base + a] - means[a]) * (samples[base + b] - means[b]);
                }
            }
        }
        let mut cov_hat = vec![0.0; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                cov_hat[a * dim + b] = accum[a * dim + b] / (nf - 1.0);
            }
        }
        // Tolérances : |ρ̂ − ρ| ≤ 0.05 sur grand échantillon (doc 9 §1.7).
        let tol = 0.05;
        for a in 0..dim {
            for b in 0..dim {
                let err = (cov_hat[a * dim + b] - cov.sigma()[a * dim + b]).abs();
                assert!(err < tol, "cov[{a}][{b}] erreur {err} > {tol}");
            }
        }
    }

    #[test]
    fn correlated_means_are_respected() {
        let cov = Covariance::diagonal(&[1.0, 1.0]).unwrap();
        let mut rng = rng();
        let samples = sample_correlated(&mut rng, &[5.0, -3.0], cov.cholesky(), 100_000).unwrap();
        let n = 100_000.0;
        let m0 = samples.iter().step_by(2).sum::<f64>() / n;
        let m1 = samples.iter().skip(1).step_by(2).sum::<f64>() / n;
        assert!((m0 - 5.0).abs() < 0.05, "m0={m0}");
        assert!((m1 + 3.0).abs() < 0.05, "m1={m1}");
    }

    #[test]
    fn sample_correlated_validates_means_length() {
        let cov = Covariance::diagonal(&[1.0, 1.0]).unwrap();
        let mut rng = rng();
        assert!(matches!(
            sample_correlated(&mut rng, &[1.0], cov.cholesky(), 10),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn pairwise_correlations_builds_valid_psd() {
        // Corrélations 3×3 valides : ρ12=0.5, ρ13=0.2, ρ23=0.1.
        let c = Covariance::from_pairwise_correlations(3, &[0.5, 0.2, 0.1]).unwrap();
        assert!(c.reconstruction_error() < 1e-12);
        assert_eq!(c.sigma()[1], 0.5);
        assert_eq!(c.sigma()[3 + 2], 0.1);
        // Mauvais nombre de corrélations.
        assert!(matches!(
            Covariance::from_pairwise_correlations(3, &[0.5]),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn pairwise_non_psd_rejected() {
        // ρ12 = 0.9, ρ13 = 0.9, ρ23 = 0.9 → matrice définie positive, OK.
        let ok = Covariance::from_pairwise_correlations(3, &[0.9, 0.9, 0.9]);
        assert!(ok.is_ok(), "{ok:?}");
        // ρ12 = −0.99, ρ13 = −0.99, ρ23 = −0.99 → non PSD (borne −1/(d−1) = −0.5).
        let bad = Covariance::from_pairwise_correlations(3, &[-0.99, -0.99, -0.99]);
        assert!(matches!(bad, Err(MathError::NotPsd(_))));
    }

    #[test]
    fn equicorrelation_blocks_are_psd() {
        let sigma = equicorrelation_matrix(&[2, 3], &[0.5, -0.2]).unwrap();
        let c = Covariance::new(sigma, 5).unwrap();
        assert!(c.reconstruction_error() < 1e-12);
        // Intra-bloc 2 : corrélation 0.5 ; inter-bloc : 0.
        assert_eq!(c.sigma()[1], 0.5);
        assert_eq!(c.sigma()[5 + 2], 0.0);
        // Bloc 3 avec ρ = −0.2 ≥ −1/2 = −0.5 : valide.
    }

    #[test]
    fn equicorrelation_rejects_psd_violation() {
        // Bloc de taille 2 : ρ = −1.5 hors [−1, 1] → paramètre invalide.
        assert!(matches!(
            equicorrelation_matrix(&[2], &[-1.5]),
            Err(MathError::InvalidParameter(_))
        ));
        // Bloc de taille 2 : ρ = −1.0 = −1/(d−1) → PSD frontière, valide.
        assert!(equicorrelation_matrix(&[2], &[-1.0]).is_ok());
        // Bloc de taille 3 : ρ = −0.6 < −0.5 → non PSD (violation de la borne).
        assert!(matches!(
            equicorrelation_matrix(&[3], &[-0.6]),
            Err(MathError::NotPsd(_))
        ));
        // Longueurs incohérentes.
        assert!(matches!(
            equicorrelation_matrix(&[2, 2], &[0.5]),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn sample_correlated_is_deterministic() {
        let cov = Covariance::new(vec![2.0, 0.5, 0.5, 1.0], 2).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        let a = sample_correlated(&mut rng_a, &[0.0, 0.0], cov.cholesky(), 1000).unwrap();
        let b = sample_correlated(&mut rng_b, &[0.0, 0.0], cov.cholesky(), 1000).unwrap();
        assert_eq!(a, b);
    }
}
