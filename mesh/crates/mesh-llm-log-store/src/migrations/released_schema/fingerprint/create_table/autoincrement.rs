#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Clause {
    Absent,
    Column(String),
}

pub(super) fn parse(tokens: &[String]) -> Option<Clause> {
    let positions = tokens
        .iter()
        .enumerate()
        .filter_map(|(position, token)| (token == "AUTOINCREMENT").then_some(position))
        .collect::<Vec<_>>();
    match positions.as_slice() {
        [] => Some(Clause::Absent),
        [4] if tokens.get(1).map(String::as_str) == Some("INTEGER")
            && tokens.get(2).map(String::as_str) == Some("PRIMARY")
            && tokens.get(3).map(String::as_str) == Some("KEY") =>
        {
            Some(Clause::Column(tokens.first()?.clone()))
        }
        _ => None,
    }
}

pub(super) fn contains_keyword(tokens: &[String]) -> bool {
    tokens.iter().any(|token| token == "AUTOINCREMENT")
}
