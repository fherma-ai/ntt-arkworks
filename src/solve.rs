//! The forward transform, through arkworks' own radix-2 domain.
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};

use crate::fherma::{Inputs, Outputs, Point, Tensor};

/// The `i`-th row of a limb tensor as a field element.
///
/// `from_bigint` refuses a value at or above the modulus. The generator
/// reduces before it writes, so a refusal here means the case is not the one
/// this specification describes — better said than folded away.
fn fe(data: &[u64], i: usize, limbs: usize) -> Fr {
    let mut words = [0u64; 4];
    words[..limbs].copy_from_slice(&data[i * limbs..(i + 1) * limbs]);
    Fr::from_bigint(BigInt::new(words)).expect("a field element below the modulus")
}

fn elements(data: &[u64], limbs: usize) -> Vec<Fr> {
    (0..data.len() / limbs).map(|i| fe(data, i, limbs)).collect()
}

fn rows(values: &[Fr], limbs: usize) -> Tensor<u64> {
    let mut data = Vec::with_capacity(values.len() * limbs);
    for v in values {
        data.extend_from_slice(&v.into_bigint().0[..limbs]);
    }
    Tensor { shape: vec![values.len() as i64, limbs as i64], data }
}

fn row(value: Fr, limbs: usize) -> Tensor<u64> {
    Tensor { shape: vec![limbs as i64], data: value.into_bigint().0[..limbs].to_vec() }
}

#[derive(Default)]
pub struct State { limbs: usize }

pub fn init(p: &Point) -> State { State { limbs: p.L as usize } }

pub fn run(state: &mut State, inp: &Inputs) -> Outputs {
    let limbs = state.limbs;
    let a = elements(&inp.a.data, limbs);
    let domain = Radix2EvaluationDomain::<Fr>::new(a.len())
        .expect("a power of two within the field's two-adicity");

    // The case names its root, and the domain picks its own from the field's
    // generator. They agree for this field — but silently computing a
    // different transform would be worse than stopping here.
    assert_eq!(domain.group_gen, fe(&inp.root.data, 0, limbs),
               "the domain's root is not the one the case names");

    Outputs { out: rows(&domain.fft(&a), limbs) }
}

pub fn free(state: State) { drop(state); }
