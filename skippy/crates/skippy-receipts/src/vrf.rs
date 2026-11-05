//! ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381), the verifiable random function
//! a client draws its challenges and checked rows from.
//!
//! The VRF gives one output per key and input, and anyone with the public
//! key can check it. So a client cannot draw a challenge again until an
//! honest node's rare outliers fail it, and a node that learns the output
//! for one input learns nothing about another. A plain Ed25519 signature
//! would not do: its signer chooses the nonce and can make many valid
//! signatures of one message.
//!
//! Only the TAI suite (try and increment) is implemented, with
//! `validate_key = TRUE`. Points are decoded strictly: an encoding must be
//! canonical, so a proof has one encoding.

use anyhow::{Result, bail};
use curve25519_dalek::edwards::{CompressedEdwardsY, EdwardsPoint};
use curve25519_dalek::scalar::{Scalar, clamp_integer};
use curve25519_dalek::traits::IsIdentity;
use sha2::{Digest, Sha512};

const SUITE: u8 = 0x03;
/// Bytes of a proof: Gamma, c and s.
pub const PROOF_LEN: usize = 32 + 16 + 32;
/// Bytes of a VRF output.
pub const OUTPUT_LEN: usize = 64;

pub type Proof = [u8; PROOF_LEN];
pub type Output = [u8; OUTPUT_LEN];

/// A VRF secret key: the 32-byte seed of RFC 8032, section 5.1.5.
#[derive(Clone)]
pub struct SecretKey {
    seed: [u8; 32],
    scalar: Scalar,
    public: [u8; 32],
}

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretKey")
            .field("public", &hex::encode(self.public))
            .finish_non_exhaustive()
    }
}

impl SecretKey {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let hashed = Sha512::digest(seed);
        let mut low = [0u8; 32];
        low.copy_from_slice(&hashed[..32]);
        let scalar = Scalar::from_bytes_mod_order(clamp_integer(low));
        let public = EdwardsPoint::mul_base(&scalar).compress().to_bytes();
        Self {
            seed,
            scalar,
            public,
        }
    }

    pub fn generate(rng: &mut impl rand::Rng) -> Self {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self::from_seed(seed)
    }

    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    pub fn public(&self) -> [u8; 32] {
        self.public
    }

    /// The proof for `alpha` (RFC 9381, section 5.1).
    pub fn prove(&self, alpha: &[u8]) -> Proof {
        let h = encode_to_curve(&self.public, alpha);
        let h_string = h.compress().to_bytes();
        let gamma = self.scalar * h;
        let hashed = Sha512::digest(self.seed);
        let k_string: [u8; 64] = Sha512::new()
            .chain_update(&hashed[32..])
            .chain_update(h_string)
            .finalize()
            .into();
        let k = Scalar::from_bytes_mod_order_wide(&k_string);
        let c = challenge(
            &self.public,
            &h,
            &gamma,
            &EdwardsPoint::mul_base(&k),
            &(k * h),
        );
        let s = k + c * self.scalar;
        let mut proof = [0u8; PROOF_LEN];
        proof[..32].copy_from_slice(&gamma.compress().to_bytes());
        proof[32..48].copy_from_slice(&c.to_bytes()[..16]);
        proof[48..].copy_from_slice(&s.to_bytes());
        proof
    }

    /// The output for `alpha`, with its proof.
    pub fn evaluate(&self, alpha: &[u8]) -> (Output, Proof) {
        let proof = self.prove(alpha);
        let gamma = decode_point(&proof[..32]).expect("a proof this key made");
        (proof_to_hash(&gamma), proof)
    }
}

/// Checks `proof` for `alpha` under `public` and returns the VRF output
/// (RFC 9381, section 5.3).
pub fn verify(public: &[u8; 32], alpha: &[u8], proof: &Proof) -> Result<Output> {
    let Some(y) = decode_point(public) else {
        bail!("the VRF key is not a point");
    };
    if y.mul_by_cofactor().is_identity() {
        bail!("the VRF key has small order");
    }
    let Some(gamma) = decode_point(&proof[..32]) else {
        bail!("the VRF proof's Gamma is not a point");
    };
    let mut c_bytes = [0u8; 32];
    c_bytes[..16].copy_from_slice(&proof[32..48]);
    let c = Scalar::from_bytes_mod_order(c_bytes);
    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&proof[48..]);
    let Some(s) = Option::<Scalar>::from(Scalar::from_canonical_bytes(s_bytes)) else {
        bail!("the VRF proof's s is not below the group order");
    };
    let h = encode_to_curve(public, alpha);
    // U = s*B - c*Y and V = s*H - c*Gamma
    let u = EdwardsPoint::vartime_double_scalar_mul_basepoint(&-c, &y, &s);
    let v = s * h - c * gamma;
    if challenge(public, &h, &gamma, &u, &v) != c {
        bail!("the VRF proof does not verify");
    }
    Ok(proof_to_hash(&gamma))
}

fn proof_to_hash(gamma: &EdwardsPoint) -> Output {
    Sha512::new()
        .chain_update([SUITE, 0x03])
        .chain_update(gamma.mul_by_cofactor().compress().to_bytes())
        .chain_update([0x00])
        .finalize()
        .into()
}

/// ECVRF_encode_to_curve_try_and_increment, with the public key as salt.
fn encode_to_curve(public: &[u8; 32], alpha: &[u8]) -> EdwardsPoint {
    for ctr in 0..=u8::MAX {
        let hash = Sha512::new()
            .chain_update([SUITE, 0x01])
            .chain_update(public)
            .chain_update(alpha)
            .chain_update([ctr, 0x00])
            .finalize();
        if let Some(point) = decode_point(&hash[..32]) {
            let point = point.mul_by_cofactor();
            if !point.is_identity() {
                return point;
            }
        }
    }
    // each attempt succeeds with probability about 1/2
    unreachable!("try and increment failed 256 times")
}

/// The challenge: the first 16 bytes of the hash of five points.
fn challenge(
    y: &[u8; 32],
    h: &EdwardsPoint,
    gamma: &EdwardsPoint,
    u: &EdwardsPoint,
    v: &EdwardsPoint,
) -> Scalar {
    let digest = Sha512::new()
        .chain_update([SUITE, 0x02])
        .chain_update(y)
        .chain_update(h.compress().to_bytes())
        .chain_update(gamma.compress().to_bytes())
        .chain_update(u.compress().to_bytes())
        .chain_update(v.compress().to_bytes())
        .chain_update([0x00])
        .finalize();
    let mut c = [0u8; 32];
    c[..16].copy_from_slice(&digest[..16]);
    Scalar::from_bytes_mod_order(c)
}

/// RFC 8032 decoding, section 5.1.3: the encoding must be canonical.
fn decode_point(bytes: &[u8]) -> Option<EdwardsPoint> {
    let compressed = CompressedEdwardsY::from_slice(bytes).ok()?;
    let point = compressed.decompress()?;
    (point.compress() == compressed).then_some(point)
}

pub fn proof_from_hex(text: &str) -> Result<Proof> {
    let bytes = hex::decode(text)?;
    Proof::try_from(bytes.as_slice())
        .map_err(|_| anyhow::anyhow!("a VRF proof has {PROOF_LEN} bytes, not {}", bytes.len()))
}

pub fn key_from_hex(text: &str) -> Result<[u8; 32]> {
    crate::commitment::hex_array(text, "a VRF key")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes<const N: usize>(text: &str) -> [u8; N] {
        hex::decode(text).unwrap().try_into().unwrap()
    }

    /// RFC 9381, appendix B.3, examples 16 to 18.
    const VECTORS: [(&str, &str, &str, &str, &str); 3] = [
        (
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "",
            "8657106690b5526245a92b003bb079ccd1a92130477671f6fc01ad16f26f723f\
             26f8a57ccaed74ee1b190bed1f479d9727d2d0f9b005a6e456a35d4fb0daab12\
             68a1b0db10836d9826a528ca76567805",
            "90cf1df3b703cce59e2a35b925d411164068269d7b2d29f3301c03dd757876ff\
             66b71dda49d2de59d03450451af026798e8f81cd2e333de5cdf4f3e140fdd8ae",
        ),
        (
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            "72",
            "f3141cd382dc42909d19ec5110469e4feae18300e94f304590abdced48aed593\
             3bf0864a62558b3ed7f2fea45c92a465301b3bbf5e3e54ddf2d935be3b67926d\
             a3ef39226bbc355bdc9850112c8f4b02",
            "eb4440665d3891d668e7e0fcaf587f1b4bd7fbfe99d0eb2211ccec90496310eb\
             5e33821bc613efb94db5e5b54c70a848a0bef4553a41befc57663b56373a5031",
        ),
        (
            "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            "af82",
            "9bc0f79119cc5604bf02d23b4caede71393cedfbb191434dd016d30177ccbf80\
             96bb474e53895c362d8628ee9f9ea3c0e52c7a5c691b6c18c9979866568add7a\
             2d41b00b05081ed0f58ee5e31b3a970e",
            "645427e5d00c62a23fb703732fa5d892940935942101e456ecca7bb217c61c45\
             2118fec1219202a0edcf038bb6373241578be7217ba85a2687f7a0310b2df19f",
        ),
    ];

    fn unwrap_hex(text: &str) -> String {
        text.split_whitespace().collect()
    }

    #[test]
    fn matches_the_rfc_vectors() {
        for (sk, pk, alpha, pi, beta) in VECTORS {
            let key = SecretKey::from_seed(bytes(sk));
            assert_eq!(hex::encode(key.public()), pk);
            let alpha = hex::decode(alpha).unwrap();
            let (output, proof) = key.evaluate(&alpha);
            assert_eq!(hex::encode(proof), unwrap_hex(pi));
            assert_eq!(hex::encode(output), unwrap_hex(beta));
            assert_eq!(verify(&bytes(pk), &alpha, &proof).unwrap(), output);
        }
    }

    #[test]
    fn a_changed_input_proof_or_key_fails() {
        let key = SecretKey::from_seed([7; 32]);
        let proof = key.prove(b"root");
        assert!(verify(&key.public(), b"root!", &proof).is_err());
        for byte in [0, 40, 70] {
            let mut bad = proof;
            bad[byte] ^= 1;
            assert!(verify(&key.public(), b"root", &bad).is_err(), "byte {byte}");
        }
        let other = SecretKey::from_seed([8; 32]);
        assert!(verify(&other.public(), b"root", &proof).is_err());
    }

    #[test]
    fn small_order_keys_are_refused() {
        // the identity point
        let mut identity = [0u8; 32];
        identity[0] = 1;
        let proof = SecretKey::from_seed([1; 32]).prove(b"x");
        assert!(verify(&identity, b"x", &proof).is_err());
    }

    #[test]
    fn a_non_canonical_s_is_refused() {
        let key = SecretKey::from_seed([3; 32]);
        let mut proof = key.prove(b"x");
        // s + q has the same value mod q but is not canonical
        let s = Scalar::from_canonical_bytes(proof[48..].try_into().unwrap()).unwrap();
        let q_bytes =
            bytes::<32>("edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010");
        let mut sum = [0u8; 32];
        let mut carry = 0u16;
        for i in 0..32 {
            let v = u16::from(s.to_bytes()[i]) + u16::from(q_bytes[i]) + carry;
            sum[i] = v as u8;
            carry = v >> 8;
        }
        if carry == 0 {
            proof[48..].copy_from_slice(&sum);
            assert!(verify(&key.public(), b"x", &proof).is_err());
        }
    }
}
