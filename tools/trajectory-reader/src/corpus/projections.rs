use super::field;
use serde_json::{Value, json};

pub(super) const SUPPORTED: &[&str] = &[
    "commitpack_edit",
    "code_refinement",
    "swe_bench_issue",
    "apps_codegen",
    "codesearchnet_explain",
    "xlam_tool_call",
    "spider_sql",
    "oasst_prompt",
    "dolly_instruction",
    "gsm8k_reasoning",
    "xsum_summarize",
    "swe_smith_trajectory_loop",
];
pub(super) struct Projection {
    pub prompt: String,
    pub expected: Value,
    pub metadata: Value,
    pub group: String,
}
pub(super) fn project(adapter: &str, row: &Value) -> Option<Projection> {
    let projection = match adapter {
        "commitpack_edit" => commitpack(row)?,
        "code_refinement" => refinement(row)?,
        "swe_bench_issue" => issue(row)?,
        "apps_codegen" => apps(row)?,
        "codesearchnet_explain" => explain(row)?,
        "xlam_tool_call" => tools(row)?,
        "spider_sql" => sql(row)?,
        "oasst_prompt" => oasst(row)?,
        "dolly_instruction" => dolly(row)?,
        "gsm8k_reasoning" => simple(
            row,
            "question",
            "answer",
            "Solve this math problem step by step.",
            "math_reasoning",
            "gsm8k:math",
        )?,
        "xsum_summarize" => simple(
            row,
            "document",
            "summary",
            "Summarize this article in one concise paragraph.",
            "summarization",
            "xsum:summarization",
        )?,
        _ => return None,
    };
    Some(projection)
}
fn make(prompt: String, expected: Value, metadata: Value, group: String) -> Projection {
    Projection {
        prompt,
        expected,
        metadata,
        group,
    }
}
fn commitpack(row: &Value) -> Option<Projection> {
    let old = field(row, "old_contents");
    let new = field(row, "new_contents");
    if old.is_empty() || new.is_empty() {
        return None;
    }
    let file = ["old_file", "new_file"]
        .iter()
        .map(|key| field(row, key))
        .find(|s| !s.is_empty())
        .unwrap_or_default();
    let repo = field(row, "repos");
    Some(make(
        format!(
            "Apply the following code change.\n\nFile: {file}\nCommit subject: {}\nCommit message:\n{}\n\nCurrent file contents:\n```{}\n{old}\n```\n\nReturn the updated file contents only.",
            field(row, "subject"),
            field(row, "message"),
            field(row, "lang")
        ),
        json!(new),
        json!({"language":row["lang"],"file":file}),
        format!(
            "commitpackft:{}",
            if repo.is_empty() { &file } else { &repo }
        ),
    ))
}
fn refinement(row: &Value) -> Option<Projection> {
    let buggy = field(row, "buggy");
    let fixed = field(row, "fixed");
    if buggy.is_empty() || fixed.is_empty() {
        return None;
    }
    Some(make(
        format!("Fix the bug in this code. Return the corrected code only.\n\n```c\n{buggy}\n```"),
        json!(fixed),
        json!({"task":"code_refinement"}),
        "codexglue:code_refinement".into(),
    ))
}
fn issue(row: &Value) -> Option<Projection> {
    let statement = field(row, "problem_statement");
    if statement.is_empty() {
        return None;
    }
    let repo = field(row, "repo");
    let hints = field(row, "hints_text");
    let mut prompt = format!(
        "You are working in repository `{repo}`.\n\nResolve this GitHub issue:\n{statement}\n"
    );
    if !hints.is_empty() {
        prompt.push_str(&format!("\nHints:\n{hints}\n"));
    }
    prompt.push_str("\nDescribe the likely code changes and tests you would make.");
    Some(make(
        prompt,
        row["patch"].clone(),
        json!({"repo":repo,"difficulty":row["difficulty"]}),
        format!("swebench:{repo}"),
    ))
}
fn apps(row: &Value) -> Option<Projection> {
    let question = field(row, "question");
    if question.is_empty() {
        return None;
    }
    let mut prompt = format!("Solve this programming problem in Python.\n\n{question}");
    let starter = field(row, "starter_code");
    if !starter.is_empty() {
        prompt.push_str(&format!("\n\nStarter code:\n```python\n{starter}\n```"));
    }
    Some(make(
        prompt,
        row["solutions"].clone(),
        json!({"difficulty":row["difficulty"]}),
        "apps:codegen".into(),
    ))
}
fn explain(row: &Value) -> Option<Projection> {
    let code = field(row, "code");
    let comment = field(row, "comment");
    if code.is_empty() || comment.is_empty() {
        return None;
    }
    Some(make(
        format!("Explain what this code does and identify any edge cases.\n\n```\n{code}\n```"),
        json!(comment),
        json!({"task":"code_explain"}),
        "codesearchnet:explain".into(),
    ))
}
fn tools(row: &Value) -> Option<Projection> {
    let query = field(row, "query");
    let tools = field(row, "tools");
    if query.is_empty() || tools.is_empty() {
        return None;
    }
    Some(make(
        format!(
            "Choose the tool call or calls needed for the user request.\nReturn only JSON.\n\nAvailable tools:\n{tools}\n\nUser request:\n{query}"
        ),
        row["answers"].clone(),
        json!({"task":"tool_call"}),
        "xlam:tool_call".into(),
    ))
}
fn sql(row: &Value) -> Option<Projection> {
    let schema = field(row, "db_schema");
    let question = field(row, "question");
    if schema.is_empty() || question.is_empty() {
        return None;
    }
    Some(make(
        format!(
            "Write a SQL query for this database schema and question.\nReturn only SQL.\n\nSchema:\n{schema}\n\nQuestion:\n{question}"
        ),
        row["query"].clone(),
        json!({"db_id":row["db_id"]}),
        format!("spider:{}", field(row, "db_id")),
    ))
}
fn oasst(row: &Value) -> Option<Projection> {
    if row["role"] != "prompter" || row["lang"] != "en" {
        return None;
    }
    let prompt = field(row, "text");
    if prompt.is_empty() {
        return None;
    }
    Some(make(
        prompt,
        Value::Null,
        json!({"lang":row["lang"]}),
        format!("oasst2:{}", field(row, "message_id")),
    ))
}
fn dolly(row: &Value) -> Option<Projection> {
    let mut prompt = field(row, "instruction");
    if prompt.is_empty() {
        return None;
    }
    let context = field(row, "context");
    if !context.is_empty() {
        prompt.push_str(&format!("\n\nContext:\n{context}"));
    }
    Some(make(
        prompt,
        row["response"].clone(),
        json!({"category":row["category"]}),
        format!("dolly:{}", field(row, "category")),
    ))
}
fn simple(
    row: &Value,
    input: &str,
    output: &str,
    instruction: &str,
    task: &str,
    group: &str,
) -> Option<Projection> {
    let body = field(row, input);
    if body.is_empty() {
        return None;
    }
    Some(make(
        format!("{instruction}\n\n{body}"),
        row[output].clone(),
        json!({"task":task}),
        group.into(),
    ))
}
