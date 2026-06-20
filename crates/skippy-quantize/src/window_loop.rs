use anyhow::Result;

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
            println!("{label}_loop_stop=max_windows completed_windows={completed}");
            return Ok(());
        }
        if !run_once()? {
            println!("{label}_loop_complete completed_windows={completed}");
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
