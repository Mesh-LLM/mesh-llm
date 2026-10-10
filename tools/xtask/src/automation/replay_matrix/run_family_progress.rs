use crate::{
    automation::replay_matrix::progress::Forwarder,
    process::{
        ObservedLine, ProbeDecision, Stream,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    },
};

pub(super) struct Owner<'a, 'scope> {
    pub launch: Option<Launch>,
    pub policy: ExpectedExit,
    pub forwarder: &'a Forwarder<'scope>,
}

impl Coordinator for Owner<'_, '_> {
    type Rejection = String;

    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<String> {
        if member == MemberId::Seed
            && line.stream == Stream::Stderr
            && let Err(error) = self.forwarder.line(line.bytes)
        {
            return ProbeDecision::Rejected(error);
        }
        ProbeDecision::Pending
    }

    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(launch) = self.launch.take() {
            return Action::StartExpected {
                launch,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::Seed
                && matches!(member.state, MemberState::ExpectedExit { .. })
        }) {
            Action::Complete
        } else {
            Action::Pending
        }
    }
}
