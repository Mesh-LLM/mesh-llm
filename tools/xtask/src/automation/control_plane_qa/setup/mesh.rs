use super::super::{coordinator::Step, http_checks::Check};
use super::builder::{Builder, Mode, Node};
use crate::command::DynResult;

pub(super) fn append(builder: &mut Builder<'_>) -> DynResult<()> {
    let options = builder.options;
    if !options.local {
        for (name, binary, offset, check_name) in [
            (
                "released-public",
                &options.released,
                0,
                "released-public-models-chat",
            ),
            (
                "current-public",
                &options.current,
                10,
                "current-public-models-chat",
            ),
        ] {
            builder.node(Node {
                name,
                binary,
                offset,
                mode: Mode::PublicClient,
            })?;
            builder.steps.push_back(Step::Check {
                name: check_name,
                check: Check::Public {
                    console: options.base + offset + 1,
                    api: options.base + offset,
                },
            });
        }
    }
    if options.local || !options.current_model.is_empty() || !options.released_model.is_empty() {
        for (server, server_binary, client, client_binary, model, offset, invite_name, pair_name) in [
            (
                "current-server",
                &options.current,
                "released-client",
                &options.released,
                &options.current_model,
                30,
                "current-server-invite",
                "current-serves-released-client",
            ),
            (
                "released-server",
                &options.released,
                "current-client",
                &options.current,
                &options.released_model,
                50,
                "released-server-invite",
                "released-serves-current-client",
            ),
        ] {
            if !options.config && model.is_empty() {
                continue;
            }
            builder.node(Node {
                name: server,
                binary: server_binary,
                offset,
                mode: Mode::Serve(model),
            })?;
            builder.steps.push_back(Step::Check {
                name: invite_name,
                check: Check::Invite {
                    console: options.base + offset + 1,
                },
            });
            builder.node(Node {
                name: client,
                binary: client_binary,
                offset: offset + 3,
                mode: Mode::JoinedClient,
            })?;
            builder.steps.push_back(Step::Check {
                name: pair_name,
                check: Check::Pair {
                    server: options.base + offset + 1,
                    client: options.base + offset + 4,
                    api: options.base + offset + 3,
                    chat: !options.config && !model.is_empty(),
                },
            });
        }
    }
    Ok(())
}
