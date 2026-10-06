use crate::command::DynResult;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use url::Url;

pub(super) fn trusted(input: &str) -> DynResult<Url> {
    if input.len() > 16384 {
        return Err("projector URL exceeds limit".into());
    }
    let mut url = Url::parse(input)?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port_or_known_default() != Some(443)
    {
        return Err("projector requires credential-free HTTPS on port 443".into());
    }
    let host = url
        .host_str()
        .ok_or("projector host absent")?
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if !["huggingface.co", "hf.co", "xethub.hf.co"]
        .iter()
        .any(|suffix| host == *suffix || host.ends_with(&format!(".{suffix}")))
    {
        return Err("untrusted projector URL host".into());
    }
    url.set_host(Some(&host))?;
    url.set_fragment(None);
    Ok(url)
}

pub(super) fn pins(host: &str, addresses: &[IpAddr]) -> DynResult<String> {
    if addresses.is_empty() || addresses.len() > 64 || addresses.iter().any(|ip| !public(*ip)) {
        return Err("projector host resolution contains absent or non-public addresses".into());
    }
    let addresses = addresses
        .iter()
        .map(|ip| match ip {
            IpAddr::V4(v) => v.to_string(),
            IpAddr::V6(v) => format!("[{v}]"),
        })
        .collect::<Vec<_>>();
    Ok(format!("{host}:443:{}", addresses.join(",")))
}

fn public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => public_v6(ip),
    }
}
fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0
        || a == 10
        || a == 127
        || a >= 224
        || a == 169 && b == 254
        || a == 172 && (16..=31).contains(&b)
        || a == 192 && b == 168
        || a == 100 && (64..=127).contains(&b)
        || a == 192 && b == 0 && (c == 0 || c == 2)
        || a == 198 && (b == 18 || b == 19 || b == 51 && c == 100)
        || a == 203 && b == 0 && c == 113
        || a == 192 && b == 88 && c == 99)
}
fn public_v6(ip: Ipv6Addr) -> bool {
    if let Some(ip) = ip.to_ipv4_mapped() {
        return public_v4(ip);
    }
    let segments = ip.segments();
    // Admit global unicast only; refuse special-use, documentation and tunnelling ranges.
    segments[0] & 0xe000 == 0x2000
        && !(segments[0] == 0x2001 && (segments[1] < 0x200 || segments[1] == 0xdb8))
        && segments[0] != 0x2002
        && !(segments[0] == 0x3fff && segments[1] < 0x1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hf_projector_policy_retains_signed_hf_suffixes_and_refuses_all_private_answers() {
        for host in ["huggingface.co", "HF.CO.", "cdn.xethub.hf.co"] {
            let url = trusted(&format!(
                "https://{host}/org/mmproj?signature=owned#fragment"
            ))
            .unwrap();
            assert_eq!(url.query(), Some("signature=owned"));
            assert_eq!(url.fragment(), None);
            assert!(pins(url.host_str().unwrap(), &["13.33.88.1".parse().unwrap()]).is_ok());
        }
        for url in [
            "http://hf.co/x",
            "https://example.com/x",
            "https://hf.co.evil/x",
            "https://u:p@hf.co/x",
            "https://hf.co:444/x",
        ] {
            assert!(trusted(url).is_err());
        }
        for ip in [
            "169.254.169.254",
            "127.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "192.0.2.1",
            "::1",
            "fc00::1",
            "::ffff:127.0.0.1",
            "2001:db8::1",
        ] {
            assert!(
                pins(
                    "hf.co",
                    &["13.33.88.1".parse().unwrap(), ip.parse().unwrap()]
                )
                .is_err(),
                "{ip}"
            );
        }
        assert!(pins("hf.co", &[]).is_err());
    }
}
