use super::{
    http_checks::{Check, MODEL},
    owner::Step,
};
use crate::{
    command::DynResult,
    process::{
        OutputFiles, ProcessSpec, Value,
        retained::{Launch, MemberId},
    },
};
use serde_json::Value as Json;
use std::{
    collections::{BTreeMap, VecDeque},
    net::{Ipv4Addr, TcpListener, UdpSocket},
    path::Path,
    time::Duration,
};
fn ports() -> DynResult<([u16; 15], Vec<TcpListener>, Vec<UdpSocket>)> {
    let mut tcp = Vec::new();
    let mut udp = Vec::new();
    let mut ports = [0; 15];
    for slot in &mut ports {
        let mut admitted = None;
        for _ in 0..64 {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
            let port = listener.local_addr()?.port();
            if let Ok(socket) = UdpSocket::bind((Ipv4Addr::LOCALHOST, port)) {
                admitted = Some((listener, socket, port));
                break;
            }
        }
        let (listener, socket, port) = admitted.ok_or("compatibility local ports unavailable")?;
        tcp.push(listener);
        udp.push(socket);
        *slot = port;
    }
    Ok((ports, tcp, udp))
}
fn launch(
    output: &Path,
    side: &Json,
    name: &str,
    provider: bool,
    join: bool,
    ports: &[u16],
    settings: (&str, Duration),
) -> DynResult<Launch> {
    let (model, readiness) = settings;
    let root = output.join(name);
    std::fs::create_dir(&root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))?;
    }
    let config = root.join("config.toml");
    let mut text = format!(
        "version = 1\n[runtime.native_runtime]\nselection = \"cpu\"\n[logging]\napplication_state_root = {}\n",
        serde_json::to_string(&root.join("logging"))?
    );
    if provider {
        text.push_str(&format!("[[models]]\nmodel = \"{MODEL}\"\n[models.hardware]\nmodel_path = {}\ngpu_layers = 0\n[models.skippy]\nsource_policy = \"local-required\"\n[models.model_fit]\nctx_size = 1024\nbatch = 128\nubatch = 64\n",serde_json::to_string(model)?));
    }
    std::fs::write(&config, text)?;
    let mut env = BTreeMap::new();
    for key in ["PATH", "SystemRoot", "WINDIR"] {
        if let Some(v) = std::env::var_os(key) {
            env.insert(key.into(), Value::Public(v));
        }
    }
    for (key, relative) in [
        ("HOME", "home"),
        ("USERPROFILE", "home"),
        ("MESH_LLM_RUNTIME_ROOT", "run"),
        ("MESH_LLM_DATA_DIR", "data"),
        ("MESH_LLM_PLUGIN_DIR", "plugins"),
        ("MESH_LLM_NODE_KEY_PATH", "node.key"),
        ("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "runtime-cache"),
    ] {
        let path = root.join(relative);
        if relative != "node.key" {
            std::fs::create_dir_all(&path)?;
        }
        env.insert(key.into(), Value::Public(path.into()));
    }
    env.insert(
        "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR".into(),
        Value::Public(
            side["runtime_root"]
                .as_str()
                .ok_or("runtime identity path")?
                .into(),
        ),
    );
    let mut args: Vec<_> = [
        if provider { "serve" } else { "client" },
        "--config",
        config.to_str().ok_or("config UTF8")?,
        "--console",
        &ports[0].to_string(),
        "--port",
        &ports[2].to_string(),
        "--bind-port",
        &ports[1].to_string(),
        "--bind-ip",
        "127.0.0.1",
        "--log-format",
        "json",
    ]
    .into_iter()
    .map(|s| Value::Public(s.into()))
    .collect();
    if provider {
        args.extend(
            [
                "--device",
                "CPU",
                "--no-draft",
                "--mesh-name",
                "PaymentCompatibility",
            ]
            .into_iter()
            .map(|s| Value::Public(s.into())),
        );
    }
    if join {
        args.extend([
            Value::Public("--join".into()),
            Value::Public("COMPATIBILITY_INVITE".into()),
        ]);
    }
    Ok(Launch {
        member: MemberId::new(name, 0)?,
        spec: ProcessSpec {
            executable: side["path"].as_str().ok_or("binary identity path")?.into(),
            arguments: args,
            environment: env,
            cwd: root.clone(),
        },
        files: OutputFiles {
            stdout: Some(root.join("stdout.log")),
            stderr: Some(root.join("stderr.log")),
        },
        readiness_deadline: readiness,
    })
}
fn append(steps: &mut VecDeque<Step>, node: Launch, console: u16, api: u16, invite: bool) {
    let id = node.member;
    steps.push_back(Step::Start(node));
    steps.push_back(Step::Check(Check::Status {
        port: console,
        invite,
    }));
    steps.push_back(Step::Admit(id));
    steps.push_back(Step::Check(Check::Models(api)));
}
pub(super) fn prepare(
    output: &Path,
    identity: &Json,
    readiness: Duration,
) -> DynResult<VecDeque<Step>> {
    let (p, _tcp, _udp) = ports()?;
    let mut steps = VecDeque::new();
    let model = identity["model"]["path"].as_str().ok_or("model path")?;
    let provider = launch(
        output,
        &identity["current"],
        "current-provider",
        true,
        false,
        &p[0..3],
        (model, readiness),
    )?;
    let first = provider.member;
    append(&mut steps, provider, p[0], p[2], true);
    let client = launch(
        output,
        &identity["released"],
        "released-client-free",
        false,
        true,
        &p[3..6],
        (model, readiness),
    )?;
    let id = client.member;
    append(&mut steps, client, p[3], p[5], false);
    steps.push_back(Step::Check(Check::Inference {
        port: p[5],
        expected: 200,
        name: "released client to current free provider",
    }));
    steps.push_back(Step::Stop(id));
    steps.push_back(Step::Check(Check::Pricing(p[0])));
    let client = launch(
        output,
        &identity["released"],
        "released-client-paid",
        false,
        true,
        &p[6..9],
        (model, readiness),
    )?;
    let id = client.member;
    append(&mut steps, client, p[6], p[8], false);
    steps.push_back(Step::Check(Check::Inference {
        port: p[8],
        expected: 402,
        name: "released client cannot bypass paid provider",
    }));
    steps.push_back(Step::Stop(id));
    steps.push_back(Step::Stop(first));
    let provider = launch(
        output,
        &identity["released"],
        "released-provider",
        true,
        false,
        &p[9..12],
        (model, readiness),
    )?;
    append(&mut steps, provider, p[9], p[11], true);
    let client = launch(
        output,
        &identity["current"],
        "current-client",
        false,
        true,
        &p[12..15],
        (model, readiness),
    )?;
    append(&mut steps, client, p[12], p[14], false);
    steps.push_back(Step::Check(Check::Inference {
        port: p[14],
        expected: 200,
        name: "current client to released free provider",
    }));
    // Guards cover selection only; release immediately before retained launches.
    Ok(steps)
}
