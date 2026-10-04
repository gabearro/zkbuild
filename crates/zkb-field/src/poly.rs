//! Small polynomial helpers (coefficient form, lowest degree first).

use crate::{Fp, Fp4};

pub fn eval(coeffs: &[Fp], x: Fp) -> Fp {
    let mut acc = Fp::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

pub fn eval_ext(coeffs: &[Fp], x: Fp4) -> Fp4 {
    let mut acc = Fp4::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

pub fn eval_ext_ext(coeffs: &[Fp4], x: Fp4) -> Fp4 {
    let mut acc = Fp4::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

/// Degree of the polynomial (index of last nonzero coefficient), or None if zero.
pub fn degree(coeffs: &[Fp]) -> Option<usize> {
    coeffs.iter().rposition(|c| !c.is_zero())
}

pub fn degree_ext(coeffs: &[Fp4]) -> Option<usize> {
    coeffs.iter().rposition(|c| !c.is_zero())
}
