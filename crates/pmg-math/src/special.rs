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

//! Fonctions spéciales partagées par les distributions.
//!
//! Implémentation interne, documentée, sans dépendance ML lourde. Toutes les
//! fonctions opèrent sur `f64` IEEE-754 et sont déterministes sur une même
//! plateforme (reproductibilité « meilleure effort » inter-plateformes, cf.
//! `docs/architecture/04-moteurs-math-injection-generation.md` §1.3.4).

/// Logarithme de la fonction Gamma, `ln Γ(x)`, pour `x > 0`.
///
/// Algorithme : approximation de Lanczos avec `g = 7`, `n = 9` coefficients.
///
/// # Entrées
/// - `x > 0` (pré-condition ; le comportement n'est pas défini sinon).
///
/// # Sorties
/// `ln Γ(x)` approché à ~1e-14 relatif.
///
/// # Complexité
/// O(1) — 9 évaluations polynomiales.
///
/// # Limites
/// Précision ~1e-14 relative ; dégradation possible pour `x` très proche de 0
/// (relation de récurrence si besoin).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Réflexion : Γ(x) = π / (sin(π x) Γ(1 − x)).
        let pi = std::f64::consts::PI;
        let s = (pi * x).sin();
        return (pi / s.abs()).ln() - ln_gamma(1.0 - x);
    }

    let xm1 = x - 1.0;
    let mut series = COEFF[0];
    for (i, c) in COEFF.iter().enumerate().skip(1) {
        series += c / (xm1 + i as f64);
    }
    let t = xm1 + G + 0.5;
    // ln Γ(x) = ½ln(2π) + (x + g − ½) ln t − t + ln(series), t = x + g − ½.
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + series.ln()
}

/// Fonction Gamma réelle `Γ(x)` pour `x > 0`.
pub fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}

/// Bêta incomplète régularisée `I_x(a, b)` pour `a, b > 0` et `0 ≤ x ≤ 1`.
///
/// Algorithme : fraction continue de Lentz (Numerical Recipes, `betacf`) avec
/// symétrie `I_x(a, b) = 1 − I_{1−x}(b, a)` pour `x` proche de 1 (convergence
/// rapide dans tous les cas).
///
/// # Entrées
/// - `a > 0`, `b > 0`, `x ∈ [0, 1]`.
///
/// # Sorties
/// `I_x(a, b) ∈ [0, 1]` (fonction de répartition de la loi Bêta).
///
/// # Complexité
/// O(itérations) avec itérations bornées (100 par défaut).
///
/// # Limites
/// Précision ~1e-12 ; renvoie 0/1 aux bornes exactes.
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x));
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    // bt = x^a (1−x)^b / B(a, b).
    let bt = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        // Symétrie : évite la lente convergence quand x → 1.
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Fraction continue de Lentz évaluant `I_x(a, b)` sans le préfacteur
/// (Numerical Recipes, §6.4, `betacf`).
///
/// # Complexité
/// O(itérations) bornées (100).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 100;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_IT {
        let m2 = 2 * m;
        // Étape paire : aa = m(b−m)x / ((a−1+2m)(a+2m)).
        let aa_even = (m as f64) * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa_even * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa_even / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Étape impaire : aa = −(a+m)(a+b+m)x / ((a+2m)(a+1+2m)).
        let aa_odd = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa_odd * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa_odd / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Gamma incomplète régularisée inférieure `P(a, x) = γ(a, x)/Γ(a)`
/// pour `a > 0`, `x ≥ 0`.
///
/// # Sorties
/// `P(a, x) ∈ [0, 1]` — fonction de répartition de la loi Gamma(shape `a`).
///
/// # Complexité
/// O(itérations) — série pour `x ≤ a+1`, fraction continue de Lentz sinon.
///
/// # Limites
/// Précision ~1e-12.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Série de la gamma incomplète inférieure (Numerical Recipes, `gser`).
fn gamma_series(a: f64, x: f64) -> f64 {
    const MAX_IT: usize = 100;
    const EPS: f64 = 3.0e-14;
    let mut sum = 1.0 / a;
    let mut term = sum;
    let mut ap = a;
    for _ in 0..MAX_IT {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Gamma incomplète supérieure `Q(a, x)` par fraction continue (NR, `gcf`).
fn gamma_cf(a: f64, x: f64) -> f64 {
    const MAX_IT: usize = 100;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = if b.abs() < FPMIN { FPMIN } else { 1.0 / b };
    let mut h = d;

    for i in 1..=MAX_IT {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Fonction de répartition du chi² : `P(χ²(ν) ≤ x)`.
///
/// Formule : `P(χ²(ν) ≤ x) = P(ν/2, x/2)` (gamma incomplète régularisée).
///
/// # Complexité
/// O(itérations de [`gamma_p`]).
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    debug_assert!(df > 0.0);
    if x <= 0.0 {
        return 0.0;
    }
    gamma_p(df / 2.0, x / 2.0)
}

/// Quantile du chi² : `x` tel que `P(χ²(ν) ≤ x) = p`.
///
/// Méthode : bissection sur [`chi2_cdf`] (robuste, bornée en itérations).
///
/// # Entrées
/// - `p ∈ (0, 1)`, `df > 0`.
///
/// # Complexité
/// O(60 · coût de [`chi2_cdf`]).
pub fn chi2_quantile(p: f64, df: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0 && df > 0.0);
    let mut lo = 0.0;
    let mut hi = df.max(1.0);
    // Double la borne haute jusqu'à dépasser le quantile.
    while chi2_cdf(hi, df) < p {
        hi *= 2.0;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if chi2_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Fonction de répartition de la loi de Student t : `P(T ≤ x)` pour `ν > 0`.
///
/// Formule (Abramowitz & Stegun, §26.7.1) : pour `x ≥ 0`,
/// `P(T ≤ x) = 1 − ½ I_{ν/(ν+x²)}(ν/2, 1/2)` ; par symétrie pour `x < 0`.
///
/// # Entrées
/// - `df > 0`, `x` réel.
///
/// # Sorties
/// Probabilité dans `[0, 1]`.
///
/// # Complexité
/// O(itérations de [`beta_inc`]) — bornée.
pub fn student_t_cdf(x: f64, df: f64) -> f64 {
    debug_assert!(df > 0.0);
    let x2 = x * x;
    let z = df / (df + x2);
    let ib = beta_inc(df / 2.0, 0.5, z);
    if x >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Quantile de la loi normale standard : `Φ⁻¹(p)` pour `p ∈ (0, 1)`.
///
/// Algorithme : approximation de Peter Acklam (rationalité de Beasley-Springer
/// et transformée rationalisée des queues), erreur relative ~1e-9.
///
/// # Entrées
/// - `p ∈ (0, 1)` (pré-condition ; `p = 0` ou `1` → ±infini).
///
/// # Sorties
/// `z` tel que `Φ(z) ≈ p`.
///
/// # Complexité
/// O(1).
pub fn normal_inv_cdf(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_672e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Queue inférieure : transformée rationalisée de la fonction Q.
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > P_HIGH {
        // Queue supérieure par symétrie.
        -normal_inv_cdf(1.0 - p)
    } else {
        // Région centrale : Beasley-Springer.
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        beta_inc, chi2_cdf, chi2_quantile, gamma, ln_gamma, normal_inv_cdf, student_t_cdf,
    };

    #[test]
    fn ln_gamma_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24.
        for (x, expected) in [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (4.0, 6.0), (5.0, 24.0)] {
            assert!((gamma(x) - expected).abs() < 1e-9, "Γ({x})");
        }
        // Γ(1/2) = √π.
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-9);
        // ln Γ(5) = ln 24.
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn beta_inc_known_values() {
        // I_x(1, 1) = x (loi uniforme sur [0,1]).
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((beta_inc(1.0, 1.0, x) - x).abs() < 1e-9, "x={x}");
        }
        // I_0.5(0.5, 0.5) = 1/2 par symétrie.
        assert!((beta_inc(0.5, 0.5, 0.5) - 0.5).abs() < 1e-9);
        // Symétrie I_x(a,b) = 1 − I_{1−x}(b,a) (convergence par les deux côtés).
        for x in [0.1, 0.3, 0.7, 0.9] {
            let l = beta_inc(2.0, 3.0, x);
            let r = 1.0 - beta_inc(3.0, 2.0, 1.0 - x);
            assert!((l - r).abs() < 1e-9, "symétrie x={x}");
        }
        // Monotonie et bornes.
        assert!(beta_inc(2.0, 3.0, 0.2) < beta_inc(2.0, 3.0, 0.8));
        assert_eq!(beta_inc(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_inc(2.0, 3.0, 1.0), 1.0);
    }

    #[test]
    fn chi2_cdf_known_values() {
        // P(χ²(2) ≤ 2) = 1 − e⁻¹ ≈ 0.63212 (chi²(2) = exponentielle de moyenne 2).
        assert!((chi2_cdf(2.0, 2.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-9);
        // P(χ²(10) ≤ 10) ≈ 0.5595 (médiane légèrement sous la moyenne).
        let v = chi2_cdf(10.0, 10.0);
        assert!((v - 0.559_506).abs() < 1e-3, "v={v}");
        // Bornes.
        assert_eq!(chi2_cdf(0.0, 5.0), 0.0);
        assert!(chi2_cdf(1000.0, 5.0) > 1.0 - 1e-9);
    }

    #[test]
    fn chi2_quantile_round_trip() {
        // Quantile → cdf → quantile : aller-retour cohérent.
        for df in [1.0, 3.0, 10.0, 30.0] {
            for p in [0.1, 0.5, 0.9] {
                let x = chi2_quantile(p, df);
                let back = chi2_cdf(x, df);
                assert!((back - p).abs() < 1e-6, "df={df} p={p} back={back}");
            }
        }
        // E[χ²(10)] = 10 : la médiane est dans une plage raisonnable.
        let med = chi2_quantile(0.5, 10.0);
        assert!((med - 9.34).abs() < 0.2, "médiane={med}");
    }

    #[test]
    fn student_t_cdf_symmetry_and_limits() {
        // Symétrie : F(x) + F(−x) = 1.
        for x in [0.5, 1.0, 2.0, 3.0] {
            let f = student_t_cdf(x, 5.0);
            let fm = student_t_cdf(-x, 5.0);
            assert!((f + fm - 1.0).abs() < 1e-9, "x={x}");
        }
        // F(0) = 0.5 ; limites 0 et 1 (queues lourdes : tolérance élargie).
        assert!((student_t_cdf(0.0, 3.0) - 0.5).abs() < 1e-12);
        assert!(student_t_cdf(-50.0, 3.0) < 1e-5);
        assert!(student_t_cdf(50.0, 3.0) > 1.0 - 1e-5);
        // df → ∞ : se rapproche de la normale.
        let f_inf = student_t_cdf(1.0, 10_000.0);
        assert!((f_inf - 0.841_344_746).abs() < 1e-3, "f_inf={f_inf}");
    }

    #[test]
    fn normal_inv_cdf_known_values() {
        // Valeurs connues : Φ⁻¹(0.5) = 0, Φ⁻¹(0.975) ≈ 1.959964, Φ⁻¹(0.01) ≈ −2.3263.
        assert!(normal_inv_cdf(0.5).abs() < 1e-9);
        assert!((normal_inv_cdf(0.975) - 1.959_964).abs() < 1e-3);
        assert!((normal_inv_cdf(0.01) + 2.326_348).abs() < 1e-2);
        // Monotonie.
        assert!(normal_inv_cdf(0.1) < normal_inv_cdf(0.5));
        assert!(normal_inv_cdf(0.5) < normal_inv_cdf(0.9));
    }
}
