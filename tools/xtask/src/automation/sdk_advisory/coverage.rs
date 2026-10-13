use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) enum Model {
    #[serde(rename = "smollm2-q8-inference")]
    Dense,
    #[serde(rename = "family-granite-hybrid")]
    Recurrent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) enum Client {
    #[serde(rename = "scripts/ci-openai-python-smoke.py")]
    Openai,
    #[serde(rename = "scripts/ci-litellm-smoke.py")]
    Litellm,
    #[serde(rename = "scripts/ci-langchain-openai-smoke.py")]
    Langchain,
}

#[derive(Debug, Serialize)]
pub(super) struct Case {
    pub(super) model: Model,
    pub(super) client: Client,
}

pub(super) fn cases() -> Vec<Case> {
    [Model::Dense, Model::Recurrent]
        .into_iter()
        .flat_map(|model| {
            [Client::Openai, Client::Litellm, Client::Langchain]
                .into_iter()
                .map(move |client| Case { model, client })
        })
        .collect()
}
