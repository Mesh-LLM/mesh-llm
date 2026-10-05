use anyhow::Result;

use crate::output::{print_info, print_success};

pub(crate) fn run_window_loop<F>(
    label: &str,
    max_windows: Option<u32>,
    mut run_once: F,
) -> Result<()>
where
    F: FnMut() -> Result<bool>,
{
    let mut completed = 0_u32;
    loop {
        if max_windows.is_some_and(|max| completed >= max) {
            print_info(format!(
                "{label} loop stopped after {completed} completed window(s)"
            ));
            return Ok(());
        }
        if !run_once()? {
            print_success(format!(
                "{label} loop complete after {completed} completed window(s)"
            ));
            return Ok(());
        }
        completed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_loop_honors_max_windows() {
        let mut calls = 0_u32;
        run_window_loop("test", Some(2), || {
            calls += 1;
            Ok(true)
        })
        .unwrap();
        assert_eq!(calls, 2);
    }
}
