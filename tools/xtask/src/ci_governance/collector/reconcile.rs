//! Refresh every bounded open main PR after policy advances, without partial lists.
use super::{GithubRead, Request, pagination, wire};
use crate::{
    ci_governance::{activation::Anchors, policy::valid_sha},
    command::DynResult,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize, PartialEq, Eq)]
struct OpenPr {
    id: u64,
    number: u64,
    state: String,
    base: Base,
    head: Head,
}
#[derive(Deserialize, PartialEq, Eq)]
struct Base {
    #[serde(rename = "ref")]
    name: String,
    sha: String,
    repo: Repository,
}
#[derive(Deserialize, PartialEq, Eq)]
struct Repository {
    id: u64,
    full_name: String,
}
#[derive(Deserialize, PartialEq, Eq)]
struct Head {
    sha: String,
}

pub(crate) fn open_pr_numbers(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    controller_sha: &str,
) -> DynResult<Vec<u64>> {
    require_main(client, anchors, controller_sha)?;
    let first = list(client, anchors, controller_sha)?;
    let second = list(client, anchors, controller_sha)?;
    require_main(client, anchors, controller_sha)?;
    if first != second {
        return Err(
            "open main PR list changed during reconciliation; no partial matrix admitted".into(),
        );
    }
    Ok(first.into_keys().collect())
}

fn require_main(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    controller_sha: &str,
) -> DynResult<()> {
    let repository: wire::Repository =
        wire::json(&pagination::read(client, &Request::Repository)?)?;
    let main: wire::MainBranch = wire::json(&pagination::read(client, &Request::CurrentMain)?)?;
    if !valid_sha(controller_sha)
        || repository.id != anchors.repository_id
        || repository.full_name != super::request::REPOSITORY
        || repository.default_branch != "main"
        || main.name != "main"
        || !main.protected
        || main.commit.sha != controller_sha
    {
        return Err(
            "reconciliation requires exact current protected main and anchored repository".into(),
        );
    }
    Ok(())
}

fn list(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    controller_sha: &str,
) -> DynResult<BTreeMap<u64, OpenPr>> {
    let mut result = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for page in 1..=3 {
        let entries: Vec<OpenPr> = wire::json(&pagination::read(
            client,
            &Request::OpenPullRequests { page },
        )?)?;
        if entries.len() > 100 || result.len() + entries.len() > 200 {
            return Err("open main PR reconciliation exceeds200PR workflow matrix bound; no partial list admitted".into());
        }
        let complete = entries.len() < 100;
        for pr in entries {
            if pr.id == 0
                || pr.number == 0
                || !ids.insert(pr.id)
                || pr.state != "open"
                || pr.base.name != "main"
                || pr.base.repo.id != anchors.repository_id
                || pr.base.repo.full_name != super::request::REPOSITORY
                || pr.base.sha != controller_sha
                || !valid_sha(&pr.head.sha)
            {
                return Err("invalid, foreign or duplicate PR in main reconciliation".into());
            }
            if result.insert(pr.number, pr).is_some() {
                return Err("duplicate PR number across reconciliation pages".into());
            }
        }
        if complete {
            return Ok(result);
        }
    }
    Err("open main PR reconciliation did not terminate completely".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_governance::collector::ApiResponse;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    const MAIN: &str = "1111111111111111111111111111111111111111";
    struct Mock {
        queue: VecDeque<(Request, Value)>,
    }
    impl GithubRead for Mock {
        fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
            let (expected, value) = self
                .queue
                .pop_front()
                .expect("unexpected reconciliation request");
            assert_eq!(*request, expected);
            Ok(ApiResponse {
                status: 200,
                body: serde_json::to_vec(&value).unwrap(),
            })
        }
    }
    fn anchors() -> Anchors {
        Anchors {
            repository_id: 1,
            reviewer_team_id: 2,
            administration_environment_id: 3,
            observer_app_id: 4,
            observer_installation_id: 5,
            publisher_app_id: 4,
            publisher_installation_id: 5,
            native_ruleset_ids: vec![8],
            human_ids: BTreeMap::from([
                ("micspiral".into(), 11),
                ("ndizazzo".into(), 12),
                ("i386".into(), 13),
            ]),
        }
    }
    fn identity(queue: &mut VecDeque<(Request, Value)>) {
        queue.push_back((
            Request::Repository,
            json!({"id":1,"full_name":"Mesh-LLM/mesh-llm","default_branch":"main"}),
        ));
        queue.push_back((
            Request::CurrentMain,
            json!({"name":"main","protected":true,"commit":{"sha":MAIN}}),
        ));
    }
    fn entry(number: u64) -> Value {
        json!({"id":number+1000,"number":number,"state":"open",
        "base":{"ref":"main","sha":MAIN,"repo":{"id":1,"full_name":"Mesh-LLM/mesh-llm"}},"head":{"sha":"2".repeat(40)}})
    }
    fn flow(count: usize) -> Mock {
        let mut queue = VecDeque::new();
        identity(&mut queue);
        for _ in 0..2 {
            let entries: Vec<_> = (1..=count).map(|number| entry(number as u64)).collect();
            for page in 1..=count / 100 + 1 {
                let start = (page - 1) * 100;
                let end = (start + 100).min(count);
                queue.push_back((
                    Request::OpenPullRequests { page: page as u32 },
                    json!(&entries[start..end]),
                ));
            }
        }
        identity(&mut queue);
        Mock { queue }
    }
    #[test]
    fn empty_exact100_and200_lists_are_complete_before_matrix_admission() {
        for count in [0, 100, 200] {
            let mut mock = flow(count);
            let numbers = open_pr_numbers(&mut mock, &anchors(), MAIN).unwrap();
            assert_eq!(numbers, (1..=count as u64).collect::<Vec<_>>());
            assert!(mock.queue.is_empty());
        }
    }
    #[test]
    fn overlimit_never_returns_a_partial_matrix() {
        let mut mock = flow(201);
        assert!(open_pr_numbers(&mut mock, &anchors(), MAIN).is_err());
    }
    #[test]
    fn duplicate_foreign_repo_wrong_base_or_state_is_rejected() {
        for (pointer, replacement) in [
            ("/base/repo/id", json!(99)),
            ("/base/repo/full_name", json!("foreign/repo")),
            ("/base/ref", json!("feature")),
            ("/state", json!("closed")),
            ("/id", json!(0)),
        ] {
            let mut mock = flow(1);
            *mock.queue[2].1[0].pointer_mut(pointer).unwrap() = replacement;
            assert!(
                open_pr_numbers(&mut mock, &anchors(), MAIN).is_err(),
                "{pointer}"
            );
        }
        let mut mock = flow(2);
        mock.queue[2].1[1] = mock.queue[2].1[0].clone();
        assert!(open_pr_numbers(&mut mock, &anchors(), MAIN).is_err());
    }
    #[test]
    fn changing_list_or_advancing_main_policy_revision_fails_closed() {
        let mut mock = flow(1);
        mock.queue[3].1[0]["head"]["sha"] = json!("3".repeat(40));
        assert!(open_pr_numbers(&mut mock, &anchors(), MAIN).is_err());
        let mut mock = flow(1);
        mock.queue.back_mut().unwrap().1["commit"]["sha"] = json!("3".repeat(40));
        assert!(open_pr_numbers(&mut mock, &anchors(), MAIN).is_err());
        let mut mock = flow(1);
        assert!(open_pr_numbers(&mut mock, &anchors(), &"3".repeat(40)).is_err());
    }
}
