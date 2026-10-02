use super::*;
fn wav(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
    let data = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let mut bytes = b"RIFF".to_vec();
    bytes.extend((36 + data.len() as u32).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16_u32.to_le_bytes());
    bytes.extend(1_u16.to_le_bytes());
    bytes.extend(channels.to_le_bytes());
    bytes.extend(rate.to_le_bytes());
    bytes.extend((rate * u32::from(channels) * 2).to_le_bytes());
    bytes.extend((channels * 2).to_le_bytes());
    bytes.extend(16_u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend((data.len() as u32).to_le_bytes());
    bytes.extend(data);
    bytes
}
#[test]
fn pcm_compares_entire_interleaved_frames_and_accepts_only_bounded_rounding_noise() {
    let reference = pcm::decode(&wav(&[1000, -1000, 1000, -1000], 8000, 2)).unwrap();
    let equal = pcm::compare(&reference, &reference).unwrap();
    assert_eq!(equal.sample_count, 2);
    assert_eq!(equal.relative_rms_error, 0.0);
    assert_eq!(equal.waveform_cosine, 1.0);
    let rounded = pcm::decode(&wav(&[1001, -999, 1001, -999], 8000, 2)).unwrap();
    assert!(pcm::compare(&rounded, &reference).is_ok());
    for samples in [
        &[1060, -1060, 1060, -1060][..],
        &[-1000, 1000, -1000, 1000],
        &[0, 0, 0, 0],
        &[1000, -1000],
    ] {
        let changed = pcm::decode(&wav(samples, 8000, 2)).unwrap();
        assert!(pcm::compare(&changed, &reference).is_err());
    }
    for (rate, channels) in [(16000, 2), (8000, 1)] {
        let changed = pcm::decode(&wav(&[1000, -1000, 1000, -1000], rate, channels)).unwrap();
        assert!(pcm::compare(&changed, &reference).is_err());
    }
}
#[test]
fn malformed_wav_cannot_supply_synthetic_pcm() {
    let good = wav(&[1000, -1000], 8000, 1);
    for bytes in [
        Vec::new(),
        good[..good.len() - 1].to_vec(),
        wav(&[], 8000, 1),
        wav(&[1000], 8000, 2),
        wav(&[1000], 0, 1),
    ] {
        assert!(pcm::decode(&bytes).is_err());
    }
    let mut nonpcm = good.clone();
    nonpcm[20] = 3;
    assert!(pcm::decode(&nonpcm).is_err());
    let mut wrongwidth = good.clone();
    wrongwidth[34] = 8;
    assert!(pcm::decode(&wrongwidth).is_err());
    let mut overflow = good;
    overflow[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(pcm::decode(&overflow).is_err());
}
#[test]
fn fresh_invocation_removes_owned_stale_wavs_and_pass_but_preserves_other_files() {
    let directory = tempfile::tempdir().unwrap();
    for name in [
        "tts-candidate.wav",
        "tts-monolithic-oracle.wav",
        "tts-oracle-result.json",
        "unrelated",
    ] {
        fs::write(directory.path().join(name), "old-pass").unwrap();
    }
    let (a, b) = execution::fresh(directory.path()).unwrap();
    assert!(!a.exists() && !b.exists());
    assert!(!directory.path().join("tts-oracle-result.json").exists());
    assert_eq!(
        fs::read_to_string(directory.path().join("unrelated")).unwrap(),
        "old-pass"
    );
}
#[test]
fn atomic_receipt_contains_measured_fields_without_partial_pending_output() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tts-oracle-result.json");
    let value = json!({"status":"pass","metrics":{"sample_rate_hz":8000,"channels":2,"sample_count":2,"relative_rms_error":0.0,"waveform_cosine":1.0}});
    receipt(&path, &value).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap()).unwrap(),
        value
    );
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}
