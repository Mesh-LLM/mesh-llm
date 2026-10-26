use super::integer::PositiveInteger;

pub(super) struct Parameters {
    pub(super) sessions_per_concurrency: PositiveInteger,
    pub(super) minimum_worker_waves: PositiveInteger,
    pub(super) minimum_context_tokens: PositiveInteger,
    pub(super) minimum_session_prompt_tokens: PositiveInteger,
    pub(super) min_isl: PositiveInteger,
    pub(super) max_isl: PositiveInteger,
    pub(super) min_turns: PositiveInteger,
    pub(super) passes: PositiveInteger,
    pub(super) warmup_turns: PositiveInteger,
    pub(super) max_output_tokens: PositiveInteger,
    pub(super) concurrency: Vec<PositiveInteger>,
}

impl Parameters {
    fn positive_fields(&self) -> [(&'static str, &PositiveInteger); 10] {
        [
            ("SESSIONS_PER_CONCURRENCY", &self.sessions_per_concurrency),
            ("MINIMUM_WORKER_WAVES", &self.minimum_worker_waves),
            ("MINIMUM_CONTEXT_TOKENS", &self.minimum_context_tokens),
            (
                "MINIMUM_SESSION_PROMPT_TOKENS",
                &self.minimum_session_prompt_tokens,
            ),
            ("MIN_ISL", &self.min_isl),
            ("MAX_ISL", &self.max_isl),
            ("MIN_TURNS", &self.min_turns),
            ("PASSES", &self.passes),
            ("WARMUP_TURNS", &self.warmup_turns),
            ("MAX_OUTPUT_TOKENS", &self.max_output_tokens),
        ]
    }

    fn concurrency(&self) -> String {
        self.concurrency
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    pub(super) fn shell_line(&self) -> String {
        let mut output = String::from("all");
        for (_, value) in self.positive_fields() {
            output.push('\t');
            output.push_str(&value.to_string());
        }
        output.push('\t');
        output.push_str(&self.concurrency());
        output.push('\n');
        output
    }

    pub(super) fn github_env(&self) -> String {
        let mut output = String::from("AGENTIC_REPLAY_MODE=all\n");
        for (name, value) in self.positive_fields() {
            output.push_str(&format!("AGENTIC_REPLAY_{name}={value}\n"));
        }
        output.push_str(&format!(
            "AGENTIC_REPLAY_CONCURRENCY={}\n",
            self.concurrency()
        ));
        output
    }
}
