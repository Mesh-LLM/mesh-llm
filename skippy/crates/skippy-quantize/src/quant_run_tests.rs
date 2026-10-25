use super::*;
use crate::splits::{ShardRange, next_missing_window_in_range};

fn parse(flags: &[&str]) -> RunQuantArgs {
    let mut argv = vec![
        "skippy-quantize",
        "run-quant",
        "--manifest",
        "missing-manifest.json",
    ];
    argv.extend_from_slice(flags);
    let crate::Command::RunQuant(args) = crate::Args::try_parse_from(argv).unwrap().command else {
        panic!("expected actual run-quant route");
    };
    args
}

#[test]
fn actual_run_quant_parser_admits_paired_range_and_refuses_incomplete_or_zero_values() {
    for flags in [
        vec!["--first-split", "2", "--last-split", "4"],
        vec!["--first-split=2", "--last-split=4"],
    ] {
        let args = parse(&flags);
        let window = args.requested_range.admit(4).unwrap();
        assert_eq!((window.first_split, window.last_split), (2, 4));
    }
    for flags in [
        vec!["--first-split", "2"],
        vec!["--last-split", "4"],
        vec!["--first-split", "0", "--last-split", "4"],
        vec!["--first-split", "1", "--last-split", "0"],
        vec!["--first-split", "1", "--last-split", "4294967296"],
    ] {
        let mut argv = vec!["skippy-quantize", "run-quant", "--manifest", "missing.json"];
        argv.extend(flags);
        assert!(crate::Args::try_parse_from(argv).is_err());
    }
}

#[test]
fn run_quant_range_bounds_and_absent_flags_preserve_internal_override_without_reading_manifest() {
    let mut args = parse(&[]);
    assert!(selected_window(&args).unwrap().is_none());
    args.window_override = Some(SplitWindow {
        first_split: 3,
        last_split: 5,
    });
    let actual = selected_window(&args).unwrap().unwrap();
    assert_eq!((actual.first_split, actual.last_split), (3, 5));
    for flags in [
        vec!["--first-split", "4", "--last-split", "2"],
        vec!["--first-split", "2", "--last-split", "5"],
    ] {
        assert!(parse(&flags).requested_range.admit(4).is_err());
    }
    let mut args = parse(&["--first-split", "2", "--last-split", "4"]);
    args.window_override = Some(SplitWindow {
        first_split: 1,
        last_split: 1,
    });
    assert!(
        selected_window(&args)
            .unwrap_err()
            .to_string()
            .contains("conflicts")
    );
}

#[test]
fn run_quant_explicit_range_resumes_inside_requested_roster_even_when_earlier_outputs_were_unlinked()
 {
    let args = parse(&["--first-split", "4", "--last-split", "6"]);
    let requested = args.requested_range.admit(8).unwrap();
    let all_missing = [ShardRange {
        first_split: 1,
        last_split: 8,
    }];
    let window = next_missing_window_in_range(&all_missing, requested).unwrap();
    assert_eq!((window.first_split, window.last_split), (4, 6));
    let partial = [
        ShardRange {
            first_split: 1,
            last_split: 3,
        },
        ShardRange {
            first_split: 5,
            last_split: 8,
        },
    ];
    let window = next_missing_window_in_range(&partial, requested).unwrap();
    assert_eq!((window.first_split, window.last_split), (5, 6));
    assert!(next_missing_window_in_range(&partial[..1], requested).is_none());
}
