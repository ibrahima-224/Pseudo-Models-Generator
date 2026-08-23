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
    use crate::low_rank::*;
    use crate::rng::DeterministicRng;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([29u8; 32])
    }

    #[test]
    fn dimensions_are_correct() {
        let mut rng = rng();
        let w = low_rank_contribution(&mut rng, 4, 5, 2, 1.0).unwrap();
        assert_eq!(w.len(), 20);
        assert!(w.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn invalid_rank_rejected() {
        let mut rng = rng();
        // rank = 0.
        assert!(matches!(
            low_rank_contribution(&mut rng, 4, 5, 0, 1.0),
            Err(MathError::InvalidRank(_))
        ));
        // rank > min(m, n).
        assert!(matches!(
            low_rank_contribution(&mut rng, 4, 5, 6, 1.0),
            Err(MathError::InvalidRank(_))
        ));
        assert!(matches!(
            low_rank_contribution(&mut rng, 2, 2, 3, 1.0),
            Err(MathError::InvalidRank(_))
        ));
    }

    #[test]
    fn rank_is_respected() {
        // m = n = 20, r = 3 : rang effectif ≈ 3 (énergie 99 %).
        let mut rng = rng();
        let w = low_rank_contribution(&mut rng, 20, 20, 3, 1.0).unwrap();
        let rank_est = effective_rank(&w, 20, 20, 0.99).unwrap();
        assert!(rank_est <= 3, "rang effectif {rank_est} > r = 3");
    }

    #[test]
    fn alpha_scales_the_contribution() {
        let mut rng_a = rng();
        let mut rng_b = rng();
        // Même seed pour U et V, alpha différent.
        let spec_a = LowRankSpec::new(2, 1.0, 1.0, 3, 4).unwrap();
        let spec_b = LowRankSpec::new(2, 2.0, 1.0, 3, 4).unwrap();
        let w1 = generate_low_rank(&mut rng_a, 3, 4, &spec_a).unwrap();
        let w2 = generate_low_rank(&mut rng_b, 3, 4, &spec_b).unwrap();
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!(
                (b - 2.0 * a).abs() < 1e-12,
                "alpha non respecté : {b} vs 2·{a}"
            );
        }
    }

    #[test]
    fn block_generation_equals_full() {
        // Streaming = complet : concaténation des blocs == produit complet.
        let spec = LowRankSpec::new(2, 1.5, 1.0, 6, 5).unwrap();
        let mut rng = rng();
        let (u, v) = generate_factors(&mut rng, 6, 5, &spec).unwrap();
        let full = low_rank_from_factors(&u, &v, 6, 5, spec.alpha);
        let mut blocks = Vec::new();
        for start in (0..6).step_by(2) {
            let end = (start + 2).min(6);
            let b = low_rank_block(&u, &v, 6, 5, start, end, spec.alpha).unwrap();
            blocks.extend(b);
        }
        assert_eq!(full, blocks, "blocs ≠ produit complet");
    }

    #[test]
    fn block_bounds_are_validated() {
        let mut rng = rng();
        let spec = LowRankSpec::new(1, 1.0, 1.0, 4, 4).unwrap();
        let (u, v) = generate_factors(&mut rng, 4, 4, &spec).unwrap();
        assert!(low_rank_block(&u, &v, 4, 4, 3, 2, 1.0).is_err());
        assert!(low_rank_block(&u, &v, 4, 4, 0, 5, 1.0).is_err());
        assert!(low_rank_block(&u, &v, 4, 4, 0, 4, 1.0).is_ok());
        // Facteurs incohérents.
        assert!(low_rank_block(&u[..3], &v, 4, 4, 0, 1, 1.0).is_err());
        assert!(low_rank_block(&u, &v[..3], 4, 4, 0, 1, 1.0).is_err());
    }

    #[test]
    fn low_rank_is_deterministic() {
        let spec = LowRankSpec::new(2, 1.0, 1.0, 4, 3).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        let a = generate_low_rank(&mut rng_a, 4, 3, &spec).unwrap();
        let b = generate_low_rank(&mut rng_b, 4, 3, &spec).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn effective_rank_of_full_rank_matrix() {
        // Matrice diagonale 4×4 de rang 4 : énergie répartie équitablement.
        let mut w = vec![0.0f64; 4 * 4];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let r = effective_rank(&w, 4, 4, 0.99).unwrap();
        assert_eq!(r, 4);
        // Matrice nulle : rang effectif 0.
        let zero = vec![0.0f64; 16];
        assert_eq!(effective_rank(&zero, 4, 4, 0.99).unwrap(), 0);
        // Rang 1 : une seule composante porte toute l'énergie.
        let mut w1 = vec![0.0f64; 4 * 4];
        for i in 0..4 {
            w1[i * 4] = 1.0;
        }
        assert_eq!(effective_rank(&w1, 4, 4, 0.99).unwrap(), 1);
    }

    #[test]
    fn effective_rank_validates_inputs() {
        assert!(matches!(
            effective_rank(&[1.0, 2.0], 2, 2, 0.99),
            Err(MathError::InvalidParameter(_))
        ));
        assert!(matches!(
            effective_rank(&[1.0; 4], 2, 2, 1.5),
            Err(MathError::InvalidParameter(_))
        ));
    }
}
