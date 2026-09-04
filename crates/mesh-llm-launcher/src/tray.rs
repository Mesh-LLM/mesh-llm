use crate::local_api;
use anyhow::Context as _;
use std::collections::HashSet;
use std::process::{Command, Stdio};
use std::time::Duration;
use tauri::AppHandle;
use tauri::Manager as _;
use tauri::menu::{CheckMenuItem, MenuBuilder, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri_plugin_autostart::ManagerExt as _;
use tauri_plugin_deep_link::DeepLinkExt as _;
use tauri_plugin_notification::NotificationExt as _;

const DEFAULT_CONSOLE_PORT: u16 = 3131;
const MAIN_WINDOW_LABEL: &str = "main";
const STARTUP_POLL_ATTEMPTS: usize = 900;
const STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(200);

pub(crate) fn run() {
    let console_port = configured_console_port(
        std::env::args().skip(1),
        std::env::var("MESH_LLM_CONSOLE_PORT").ok(),
    );
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(
            move |app, arguments, _cwd| {
                let mut opened_pairing = false;
                for argument in arguments {
                    opened_pairing |= handle_pairing_link(app, &argument, console_port);
                }
                if !opened_pairing {
                    open_app_route(app, &console_url(console_port), console_port);
                }
            },
        ))
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| setup(app, console_port))
        .on_window_event(|window, event| {
            if window.label() == MAIN_WINDOW_LABEL
                && let tauri::WindowEvent::CloseRequested { api, .. } = event
            {
                api.prevent_close();
                let _ = window.hide();
            }
        });
    let app = builder
        .build(tauri::generate_context!())
        .expect("Mesh could not start");
    app.run(move |app, event| {
        #[cfg(target_os = "macos")]
        if let tauri::RunEvent::Reopen {
            has_visible_windows: false,
            ..
        } = event
        {
            open_app_route(app, &console_url(console_port), console_port);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (app, event, console_port);
    });
}

fn setup(app: &mut tauri::App, console_port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let status = MenuItem::with_id(app, "status", "Mesh checking…", false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", "Open Mesh", true, None::<&str>)?;
    let pair = MenuItem::with_id(app, "pair", "Pair a device…", true, None::<&str>)?;
    let pending = MenuItem::with_id(
        app,
        "pending",
        "No connection requests",
        false,
        None::<&str>,
    )?;
    let start = MenuItem::with_id(app, "start", "Start Mesh", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop", "Stop Mesh", false, None::<&str>)?;
    let start_at_login = CheckMenuItem::with_id(
        app,
        "start-at-login",
        "Start Mesh at login",
        true,
        app.autolaunch().is_enabled().unwrap_or(false),
        None::<&str>,
    )?;
    let diagnostics =
        MenuItem::with_id(app, "diagnostics", "Logs & diagnostics", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Mesh", true, None::<&str>)?;
    let menu = MenuBuilder::new(app)
        .item(&status)
        .separator()
        .item(&open)
        .item(&pair)
        .item(&pending)
        .separator()
        .item(&start)
        .item(&stop)
        .item(&start_at_login)
        .separator()
        .item(&diagnostics)
        .item(&quit)
        .build()?;

    let start_for_menu = start.clone();
    let stop_for_menu = stop.clone();
    let login_for_menu = start_at_login.clone();
    let icon = app
        .default_window_icon()
        .cloned()
        .ok_or("Mesh bundle icon is missing")?;
    TrayIconBuilder::new()
        .icon(icon)
        .icon_as_template(true)
        .tooltip("Mesh")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "open" => open_app_route(app, &console_url(console_port), console_port),
            "pair" | "pending" => open_app_route(app, &pairing_url(console_port), console_port),
            "start" => match start_mesh(console_port) {
                Ok(()) => {
                    let _ = start_for_menu.set_enabled(false);
                    let _ = stop_for_menu.set_enabled(true);
                }
                Err(error) => notify_start_error(app, &error),
            },
            "stop" => {
                let _ = local_api::shutdown_mesh(console_port);
            }
            "start-at-login" => {
                // Native check-menu implementations toggle before dispatching the event.
                let enabled = login_for_menu.is_checked().unwrap_or(false);
                let result = if enabled {
                    app.autolaunch().enable()
                } else {
                    app.autolaunch().disable()
                };
                if result.is_err() {
                    let _ = login_for_menu.set_checked(!enabled);
                }
            }
            "diagnostics" => open_app_route(app, &diagnostics_url(console_port), console_port),
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;

    let opened_deep_link = install_deep_link_handlers(app, console_port)?;
    start_status_monitor(
        app.handle().clone(),
        status,
        pending,
        start,
        stop,
        console_port,
    );
    if !opened_deep_link {
        open_app_route(app.handle(), &console_url(console_port), console_port);
    }
    Ok(())
}

fn install_deep_link_handlers(
    app: &tauri::App,
    console_port: u16,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut opened_deep_link = false;
    if let Some(urls) = app.deep_link().get_current()? {
        for url in urls {
            opened_deep_link |= handle_pairing_link(app.handle(), url.as_str(), console_port);
        }
    }
    let handle = app.handle().clone();
    app.deep_link().on_open_url(move |event| {
        for url in event.urls() {
            handle_pairing_link(&handle, url.as_str(), console_port);
        }
    });
    Ok(opened_deep_link)
}

fn handle_pairing_link(app: &AppHandle, value: &str, console_port: u16) -> bool {
    if let Some(url) = pairing_console_url(value, console_port) {
        open_app_route(app, &url, console_port);
        true
    } else {
        false
    }
}

fn pairing_console_url(value: &str, console_port: u16) -> Option<String> {
    let Ok(url) = url::Url::parse(value) else {
        return None;
    };
    if url.scheme() != "mesh-llm" || url.host_str() != Some("pair") {
        return None;
    }
    let offer = url.path().trim_matches('/');
    if offer.is_empty()
        || offer.len() > 16 * 1024
        || !offer
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return None;
    }
    Some(format!(
        "{}?pair={offer}#pairing",
        console_url(console_port)
    ))
}

fn show_app_window(app: &AppHandle, url: &str) -> anyhow::Result<()> {
    let window = app
        .get_webview_window(MAIN_WINDOW_LABEL)
        .context("the Mesh application window is unavailable")?;
    window.navigate(url::Url::parse(url).context("invalid Mesh application route")?)?;
    window.show()?;
    window.set_focus()?;
    Ok(())
}

fn open_app_route(app: &AppHandle, url: &str, console_port: u16) {
    if local_api::mesh_peer_count(console_port).is_some() {
        if let Err(error) = show_app_window(app, url) {
            notify_start_error(app, &error);
        }
        return;
    }
    if let Err(error) = start_mesh(console_port) {
        notify_start_error(app, &error);
        return;
    }
    let app = app.clone();
    let url = url.to_string();
    std::thread::spawn(move || {
        for _ in 0..STARTUP_POLL_ATTEMPTS {
            if local_api::mesh_peer_count(console_port).is_some() {
                let window_app = app.clone();
                let _ = app.run_on_main_thread(move || {
                    if let Err(error) = show_app_window(&window_app, &url) {
                        notify_start_error(&window_app, &error);
                    }
                });
                return;
            }
            std::thread::sleep(STARTUP_POLL_INTERVAL);
        }
        show_startup_failure(
            &app,
            "Mesh took too long to start. Quit Mesh, check the runtime installation, and try again.",
        );
        let _ = app
            .notification()
            .builder()
            .title("Mesh did not start")
            .body("Quit Mesh, check the runtime installation, and try again.")
            .show();
    });
}

fn notify_start_error(app: &AppHandle, error: &anyhow::Error) {
    show_startup_failure(app, &format!("Mesh could not start: {error:#}"));
    let _ = app
        .notification()
        .builder()
        .title("Mesh could not start")
        .body(format!("{error:#}"))
        .show();
}

fn show_startup_failure(app: &AppHandle, message: &str) {
    let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) else {
        return;
    };
    let Ok(message) = serde_json::to_string(message) else {
        return;
    };
    let _ = window.eval(format!("window.meshLauncherFailure?.({message})"));
    let _ = window.show();
    let _ = window.set_focus();
}

fn start_mesh(console_port: u16) -> anyhow::Result<()> {
    if local_api::mesh_peer_count(console_port).is_some() {
        return Ok(());
    }
    let binary = std::env::var_os("MESH_LLM_BIN")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|path| path.parent().map(|parent| parent.join(mesh_binary_name())))
                .filter(|path| path.is_file())
        })
        .or_else(|| std::env::current_exe().ok())
        .unwrap_or_else(|| std::path::PathBuf::from(mesh_binary_name()));
    Command::new(binary)
        .arg("serve")
        .arg("--console")
        .arg(console_port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("the bundled Mesh runtime could not be launched")?;
    Ok(())
}

fn mesh_binary_name() -> &'static str {
    if cfg!(windows) {
        "mesh-llm.exe"
    } else {
        "mesh-llm"
    }
}

fn configured_console_port(
    arguments: impl IntoIterator<Item = String>,
    environment: Option<String>,
) -> u16 {
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let value = argument
            .strip_prefix("--console=")
            .map(str::to_owned)
            .or_else(|| {
                (argument == "--console")
                    .then(|| arguments.next())
                    .flatten()
            });
        if let Some(port) = value.and_then(|value| value.parse::<u16>().ok())
            && port != 0
        {
            return port;
        }
    }
    environment
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port != 0)
        .unwrap_or(DEFAULT_CONSOLE_PORT)
}

fn console_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/")
}

fn pairing_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/#pairing")
}

fn diagnostics_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/logs")
}

fn start_status_monitor(
    app: AppHandle,
    status_item: MenuItem<tauri::Wry>,
    pending_item: MenuItem<tauri::Wry>,
    start_item: MenuItem<tauri::Wry>,
    stop_item: MenuItem<tauri::Wry>,
    console_port: u16,
) {
    std::thread::spawn(move || {
        let mut notified = HashSet::new();
        loop {
            let peers = local_api::mesh_peer_count(console_port);
            let running = peers.is_some();
            let _ = status_item.set_text(match peers {
                Some(1) => "Mesh running · 1 peer".to_string(),
                Some(count) => format!("Mesh running · {count} peers"),
                None => "Mesh stopped".to_string(),
            });
            let _ = start_item.set_enabled(!running);
            let _ = stop_item.set_enabled(running);

            let pending = local_api::pairing_sessions(console_port)
                .into_iter()
                .filter(|session| {
                    session.direction == "incoming" && session.status == "awaiting_approval"
                })
                .collect::<Vec<_>>();
            let _ = pending_item.set_text(match pending.len() {
                0 => "No connection requests".to_string(),
                1 => "1 connection request…".to_string(),
                count => format!("{count} connection requests…"),
            });
            let _ = pending_item.set_enabled(!pending.is_empty());
            for session in pending {
                if notified.insert(session.id) {
                    let _ = app
                        .notification()
                        .builder()
                        .title("Mesh connection request")
                        .body(format!(
                            "{} wants to connect. Open Mesh to compare codes.",
                            session.peer_name
                        ))
                        .show();
                }
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_binary_name_matches_platform() {
        assert_eq!(mesh_binary_name().ends_with(".exe"), cfg!(windows));
    }

    #[test]
    fn rejects_non_pairing_deep_links_before_opening() {
        assert_eq!(
            pairing_console_url("https://example.com/pair/value", 3131),
            None
        );
        assert_eq!(
            pairing_console_url("mesh-llm://pair/contains%20space", 3131),
            None
        );
        assert_eq!(pairing_console_url("mesh-llm://other/value", 3131), None);
    }

    #[test]
    fn maps_valid_pairing_deep_links_to_the_web_console() {
        assert_eq!(
            pairing_console_url("mesh-llm://pair/abc_123-def", 4242),
            Some("http://127.0.0.1:4242/?pair=abc_123-def#pairing".to_string())
        );
    }

    #[test]
    fn console_port_accepts_arguments_then_environment_then_default() {
        assert_eq!(
            configured_console_port(["--console".into(), "4242".into()], Some("5252".into())),
            4242
        );
        assert_eq!(
            configured_console_port(["--console=4343".into()], Some("5252".into())),
            4343
        );
        assert_eq!(configured_console_port([], Some("5252".into())), 5252);
        assert_eq!(configured_console_port([], Some("invalid".into())), 3131);
    }
}
