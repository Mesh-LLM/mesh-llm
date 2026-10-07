use anyhow::Result;
use skippy_model_ref::package_reference::PackageReference;

pub(crate) fn write(reference: &str, output: &mut dyn std::io::Write) -> Result<()> {
    let parsed = PackageReference::parse(reference)?;
    writeln!(output, "{}\n{}", parsed.repo(), parsed.revision())?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lines_are_exact_and_invalid_input_has_no_output() {
        let mut output = Vec::new();
        write("hf://a/b:topic/one", &mut output).unwrap();
        assert_eq!(output, b"a/b\ntopic/one\n");
        output.clear();
        assert!(write("hf://a/b@../one", &mut output).is_err());
        assert!(output.is_empty());
    }
    struct RefusingWriter {
        refuse_flush: bool,
    }
    impl std::io::Write for RefusingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.refuse_flush {
                Ok(bytes.len())
            } else {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
    }
    #[test]
    fn write_and_flush_refusals_propagate() {
        for refuse_flush in [false, true] {
            assert!(write("hf://a/b", &mut RefusingWriter { refuse_flush }).is_err());
        }
    }
}
