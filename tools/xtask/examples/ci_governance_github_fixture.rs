//! Inert finite GitHub transport fixture. Never contacts a network endpoint.
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};

fn main() {
    if let Err(error) = run() {
        eprintln!("inert governance transport refused: {error}");
        std::process::exit(9);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(!args.iter().any(|arg| arg.contains("fixture-jwt")
        || arg.contains("fixture-observer")
        || arg.contains("fixture-publisher")
        || arg.contains("PRIVATE KEY")));
    for name in [
        "CI_GOVERNANCE_APP_PRIVATE_KEY",
        "CI_GOVERNANCE_APP_JWT",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "credential source must not reach transport child"
        );
    }
    let fixture: Value = serde_json::from_slice(&fs::read(".governance-fixture.json")?)?;
    let curl = args.first().map(String::as_str) == Some("--disable");
    let (method, endpoint, body) = if curl {
        assert!(std::env::var_os("GH_TOKEN").is_none());
        let header = args
            .windows(2)
            .find(|pair| pair[0] == "--header" && pair[1].starts_with('@'))
            .unwrap()[1]
            .trim_start_matches('@');
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(header)?.permissions().mode() & 0o777, 0o600);
        }
        let auth = fs::read_to_string(header)?;
        assert!(auth.starts_with("Authorization: Bearer "));
        assert!(auth.contains("fixture-jwt") || auth.split('.').count() == 3);
        append("jwt-header-path", json!(header))?;
        let method = value(&args, "--request");
        let endpoint = value(&args, "--url")
            .strip_prefix("https://api.github.com/")
            .unwrap()
            .to_owned();
        let body = args
            .windows(2)
            .find(|pair| pair[0] == "--data-binary")
            .map(|pair| read_json(pair[1].trim_start_matches('@')))
            .transpose()?
            .unwrap_or(Value::Null);
        (method.to_owned(), endpoint, body)
    } else {
        assert_eq!(args.first().map(String::as_str), Some("api"));
        assert_eq!(value(&args, "--hostname"), "github.com");
        let token = std::env::var("GH_TOKEN")?;
        assert!(token == "fixture-observer" || token == "fixture-publisher");
        let position = args.iter().position(|arg| arg == "--method").unwrap();
        let body = args
            .windows(2)
            .find(|pair| pair[0] == "--input")
            .map(|pair| read_json(&pair[1]))
            .transpose()?
            .unwrap_or(Value::Null);
        (args[position + 1].clone(), args[position + 2].clone(), body)
    };
    append(
        "request",
        json!({"method":method,"endpoint":endpoint,"body":body}),
    )?;
    let (status, reply) = if curl {
        app(&fixture, &method, &endpoint, &body)?
    } else {
        repository(&fixture, &method, &endpoint, &body)?
    };
    print!("HTTP/2 {status} Fixture\r\ncontent-type: application/json\r\n\r\n");
    std::io::stdout().write_all(&reply)?;
    Ok(())
}
fn value<'a>(args: &'a [String], flag: &str) -> &'a str {
    &args.windows(2).find(|pair| pair[0] == flag).unwrap()[1]
}
fn read_json(path: impl AsRef<Path>) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn append(kind: &str, value: Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(".governance-calls.jsonl")?;
    writeln!(file, "{}", json!({"kind":kind,"value":value}))?;
    Ok(())
}
fn permissions(checks: &str) -> Value {
    json!({"actions":"read","administration":"read","checks":checks,"contents":"read","environments":"read","members":"read","metadata":"read","pull_requests":"read"})
}
fn app(
    fixture: &Value,
    method: &str,
    endpoint: &str,
    body: &Value,
) -> Result<(u16, Vec<u8>), Box<dyn std::error::Error>> {
    let result = match (method, endpoint) {
        ("GET", "app") => json!({"id":4,"permissions":permissions("write")}),
        ("GET", "app/installations/5") => {
            json!({"id":5,"app_id":4,"target_type":"Organization","account":{"login":"Mesh-LLM"},"permissions":permissions("write"),"repository_selection":"selected","suspended_at":null})
        }
        ("POST", "app/installations/5/access_tokens") => {
            assert_eq!(body["repository_ids"], json!([1]));
            let publisher = body["permissions"]["checks"] == "write";
            let expected = if publisher {
                json!({"checks":"write","metadata":"read"})
            } else {
                permissions("read")
            };
            assert_eq!(body["permissions"], expected);
            let mut grant = expected;
            if fixture["mode"] == "wide-token" {
                grant["administration"] = json!("write");
            }
            return Ok((
                201,
                serde_json::to_vec(
                    &json!({"token":if publisher {"fixture-publisher"}else{"fixture-observer"},"expires_at":fixture["expires_at"],"permissions":grant,"repositories":[{"id":1,"full_name":"Mesh-LLM/mesh-llm"}],"repository_selection":"selected"}),
                )?,
            ));
        }
        _ => return Err("unexpected fixed App request".into()),
    };
    Ok((200, serde_json::to_vec(&result)?))
}
fn actor(login: &str, id: u64) -> Value {
    json!({"__typename":"User","login":login,"databaseId":id})
}
fn repository(
    fixture: &Value,
    method: &str,
    endpoint: &str,
    body: &Value,
) -> Result<(u16, Vec<u8>), Box<dyn std::error::Error>> {
    const REPO: &str = "repos/Mesh-LLM/mesh-llm";
    let main = &fixture["main_sha"];
    let head = "2".repeat(40);
    let result = if endpoint == REPO {
        json!({"id":1,"full_name":"Mesh-LLM/mesh-llm","default_branch":"main"})
    } else if endpoint == format!("{REPO}/branches/main") {
        json!({"name":"main","protected":true,"commit":{"sha":main}})
    } else if endpoint.contains("/contents/ci/governance/policy.json") {
        return Ok((200, fixture["policy"].as_str().unwrap().as_bytes().to_vec()));
    } else if endpoint.contains("/pulls/") && !endpoint.contains("/files?") {
        let number = endpoint.rsplit('/').next().unwrap().parse::<u64>()?;
        json!({"number":number,"state":"open","user":{"id":99,"login":"contributor","type":"User"},"base":{"ref":"main","sha":main,"repo":{"id":1,"full_name":"Mesh-LLM/mesh-llm"}},"head":{"sha":head},"changed_files":1})
    } else if endpoint.contains("/files?") {
        json!([{"filename":"ci/governance/policy.json","status":"modified","previous_filename":null}])
    } else if endpoint == "orgs/Mesh-LLM/teams/ci-policy-reviewers" {
        json!({"id":2,"slug":"ci-policy-reviewers","privacy":"closed","organization":{"login":"Mesh-LLM"},"parent":null})
    } else if endpoint.ends_with("teams/ci-policy-reviewers/repos/Mesh-LLM/mesh-llm") {
        json!({"id":1,"full_name":"Mesh-LLM/mesh-llm","permissions":{"push":true,"pull":true,"admin":false,"maintain":false},"role_name":"write"})
    } else if endpoint == "graphql" {
        if body["query"].as_str().unwrap().contains("reviews(") {
            let connection = json!({"totalCount":1,"nodes":[{"fullDatabaseId":"10","state":if fixture["mode"]=="denied"{"DISMISSED"}else{"APPROVED"},"submittedAt":"2023-11-14T22:13:20Z","commit":{"oid":head},"author":actor("micspiral",11)}],"pageInfo":{"hasNextPage":false,"endCursor":null}});
            json!({"data":{"repository":{"pullRequest":{"reviews":connection}}}})
        } else {
            let connection = json!({"totalCount":3,"nodes":[actor("micspiral",11),actor("ndizazzo",12),actor("i386",13)],"pageInfo":{"hasNextPage":false,"endCursor":null}});
            json!({"data":{"organization":{"team":{"databaseId":2,"members":connection}}}})
        }
    } else if endpoint == format!("{REPO}/check-runs") {
        assert_eq!(method, "POST");
        assert_eq!(std::env::var("GH_TOKEN")?, "fixture-publisher");
        assert_eq!(body["name"], "CI policy / independent approval");
        assert_eq!(body["status"], "in_progress");
        let mut check = body.clone();
        check["id"] = json!(7);
        check["app"] = json!({"id":4});
        check["conclusion"] = Value::Null;
        match fixture["mode"].as_str().unwrap() {
            "wrong-app" => check["app"]["id"] = json!(15368),
            "wrong-head" => check["head_sha"] = json!("3".repeat(40)),
            "wrong-external" => check["external_id"] = json!("foreign"),
            _ => {}
        }
        fs::write(".governance-check.json", serde_json::to_vec(&check)?)?;
        return Ok((201, serde_json::to_vec(&check)?));
    } else if endpoint == format!("{REPO}/check-runs/7") {
        assert_eq!(std::env::var("GH_TOKEN")?, "fixture-publisher");
        let mut check = read_json(".governance-check.json")?;
        if method == "PATCH" {
            check["status"] = body["status"].clone();
            check["conclusion"] = body["conclusion"].clone();
            fs::write(".governance-check.json", serde_json::to_vec(&check)?)?;
        }
        check
    } else if endpoint.ends_with("/actions/workflows/ci-governance-review-signal.yml") {
        json!({"id":9,"path":".github/workflows/ci-governance-review-signal.yml","state":"active"})
    } else if endpoint.ends_with("/actions/runs/500") {
        json!({"id":500,"workflow_id":9,"path":"Mesh-LLM/mesh-llm/.github/workflows/ci-governance-review-signal.yml@refs/pull/2781/merge","event":"pull_request_review","status":"completed","repository":{"id":1},"pull_requests":null,"head_branch":"feature","head_repository":{"id":44,"owner":{"login":"contributor"}}})
    } else if endpoint.contains("/pulls?state=open") {
        json!([{"id":100,"number":if fixture["mode"]=="retarget" {2782}else{2781},"state":"open","user":{"id":99,"login":"contributor","type":"User"},"changed_files":1,"head":{"sha":head,"ref":"feature","repo":{"id":44,"owner":{"login":"contributor"}}},"base":{"ref":"main","sha":main,"repo":{"id":1,"full_name":"Mesh-LLM/mesh-llm"}}}])
    } else {
        return Err("unexpected fixed observer/publisher request".into());
    };
    Ok((200, serde_json::to_vec(&result)?))
}
