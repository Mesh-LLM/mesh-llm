use super::{Error, Generated};
use crate::automation::codepoint_json::{emission::write_string, strings::JsonString};

pub(super) fn encode(generated: &Generated) -> Result<String, Error> {
    let mut output = String::from("{\"first_summary\": ");
    summary(&mut output, &generated.first_summary);
    output.push_str(", \"output\": ");
    let path = generated
        .output
        .to_str()
        .ok_or(Error::Arguments("output path must be UTF-8"))?;
    write_string(&mut output, &JsonString::from(path));
    output.push_str(", \"second_summary\": ");
    summary(&mut output, &generated.second_summary);
    output.push_str(&format!(", \"shards\": {}}}", generated.shards));
    Ok(output)
}

fn summary(output: &mut String, counts: &crate::automation::rewriter_report::generator::Summary) {
    output.push('{');
    for (index, (key, count)) in counts.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write_string(output, &JsonString::from(*key));
        output.push_str(&format!(": {count}"));
    }
    output.push('}');
}
