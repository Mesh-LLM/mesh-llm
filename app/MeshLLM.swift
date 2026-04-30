// MeshLLM — macOS menu-bar launcher
//
// Build:
//   swiftc -parse-as-library -O -framework SwiftUI -framework AppKit \
//       -o MeshLLM.app/Contents/MacOS/MeshLLM MeshLLM.swift
//

import SwiftUI

// MARK: - Config

private let kPollInterval: TimeInterval = 3
private let kAPIBase = "http://localhost:3131"
private let kConfigDir = NSString("~/.config/mesh-llm-app").expandingTildeInPath
private let kTokenFile = (kConfigDir as NSString).appendingPathComponent("token")

// MARK: - Status

struct MeshStatus: Codable {
    let version: String
    let node_state: String
    let node_status: String
    let llama_ready: Bool
    let model_name: String?
    let serving_models: [String]
    let api_port: Int?
    let my_vram_gb: Double?
    let peers: [Peer]
    let mesh_id: String?
    let token: String?
    let nostr_discovery: Bool?

    struct Peer: Codable {
        let serving_models: [String]
    }
}

// MARK: - Manager

@MainActor
final class MeshManager: ObservableObject {
    @Published var status: MeshStatus?
    @Published var isRunning = false
    @Published var isStarting = false
    @Published var joinToken = ""

    private var pollTimer: Timer?

    var binaryPath: String {
        let candidates = [
            "\(NSHomeDirectory())/.local/bin/mesh-llm",
            "/usr/local/bin/mesh-llm",
            "/opt/homebrew/bin/mesh-llm",
        ]
        for p in candidates {
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return candidates[0]
    }

    var savedToken: String? {
        guard FileManager.default.fileExists(atPath: kTokenFile),
              let t = try? String(contentsOfFile: kTokenFile, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !t.isEmpty
        else { return nil }
        return t
    }

    func saveToken(_ t: String) {
        try? FileManager.default.createDirectory(atPath: kConfigDir, withIntermediateDirectories: true)
        try? t.trimmingCharacters(in: .whitespacesAndNewlines)
            .write(toFile: kTokenFile, atomically: true, encoding: .utf8)
    }

    func clearToken() {
        try? FileManager.default.removeItem(atPath: kTokenFile)
        joinToken = ""
    }

    func boot() {
        joinToken = savedToken ?? ""
        startPolling()
    }

    func startPolling() {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: kPollInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in await self?.poll() }
        }
        Task { await poll() }
    }

    func poll() async {
        guard let url = URL(string: "\(kAPIBase)/api/status") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let s = try JSONDecoder().decode(MeshStatus.self, from: data)
            status = s
            isRunning = true
            isStarting = false
        } catch {
            status = nil
            isRunning = false
        }
    }

    /// Start with public auto-discovery
    func startPublic() {
        launch(args: "serve --auto")
    }

    /// Start private mesh (no discovery)
    func startPrivate() {
        launch(args: "serve")
    }

    /// Join an existing mesh with a token
    func startWithToken(_ token: String) {
        let t = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        saveToken(t)
        let escaped = t.replacingOccurrences(of: "'", with: "'\\''")
        launch(args: "serve --join '\(escaped)'")
    }

    private func launch(args: String) {
        guard !isRunning && !isStarting else { return }
        isStarting = true
        let cmd = "nohup '\(binaryPath)' \(args) > /dev/null 2>&1 & disown"
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/sh")
        proc.arguments = ["-c", cmd]
        try? proc.run()
        proc.waitUntilExit()
    }

    func stop() {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binaryPath)
        proc.arguments = ["stop"]
        try? proc.run()
        proc.waitUntilExit()
        isRunning = false
        status = nil
    }

    func openChat() {
        if let url = URL(string: "http://localhost:3131") {
            NSWorkspace.shared.open(url)
        }
    }

    func copyToken() {
        guard let token = status?.token, !token.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(token, forType: .string)
    }

    var meshModelCount: Int {
        guard let s = status else { return 0 }
        let peerModels = s.peers.flatMap { $0.serving_models }
        return Set(s.serving_models + peerModels).count
    }

    var peerCount: Int { status?.peers.count ?? 0 }

    var isPrivateMesh: Bool { status?.nostr_discovery == false }
}

// MARK: - Join Window

final class JoinWindowController: NSWindowController {
    static var shared: JoinWindowController?

    static func show(manager: MeshManager) {
        if let existing = shared {
            existing.window?.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let view = JoinView(manager: manager)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Join Mesh"
        window.styleMask = [.titled, .closable]
        window.setContentSize(NSSize(width: 400, height: 150))
        window.center()
        window.isReleasedWhenClosed = false

        let controller = JoinWindowController(window: window)
        shared = controller
        controller.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    static func close() {
        shared?.window?.close()
        shared = nil
    }
}

struct JoinView: View {
    @ObservedObject var manager: MeshManager
    @State private var token = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Paste a join token from someone sharing their mesh.")
                .font(.caption)
                .foregroundStyle(.secondary)

            TextField("Join token", text: $token)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
                .onSubmit { go() }
                .onAppear { token = manager.joinToken }

            HStack {
                Spacer()
                Button("Cancel") { JoinWindowController.close() }
                    .keyboardShortcut(.cancelAction)
                Button("Join") { go() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(token.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding()
    }

    func go() {
        let t = token.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        manager.startWithToken(t)
        JoinWindowController.close()
    }
}

// MARK: - Menu Bar Icon

private func makeMenuBarIcon() -> NSImage {
    let size = NSSize(width: 18, height: 18)
    let img = NSImage(size: size, flipped: false) { _ in
        NSColor.black.setFill()
        NSColor.black.setStroke()

        NSBezierPath(ovalIn: NSRect(x: 2.5, y: 9, width: 13, height: 8)).fill()

        let xs: [CGFloat] = [4.5, 6.5, 9, 11.5, 13.5]
        for (i, x) in xs.enumerated() {
            let p = NSBezierPath()
            p.lineWidth = 1.2
            p.lineCapStyle = .round
            let endY: CGFloat = i % 2 == 0 ? 1.5 : 2.5
            let c: CGFloat = i % 2 == 0 ? -1.5 : 1.5
            p.move(to: NSPoint(x: x, y: 9.5))
            p.curve(to: NSPoint(x: x + c * 0.5, y: endY),
                     controlPoint1: NSPoint(x: x + c, y: 7),
                     controlPoint2: NSPoint(x: x - c * 0.5, y: 4))
            p.stroke()
        }
        return true
    }
    img.isTemplate = true
    return img
}

// MARK: - App

@main
struct MeshLLMApp: App {
    @StateObject private var manager = MeshManager()

    var body: some Scene {
        MenuBarExtra {
            MenuContent(manager: manager)
        } label: {
            Image(nsImage: makeMenuBarIcon())
        }
    }
}

// MARK: - Menu

struct MenuContent: View {
    @ObservedObject var manager: MeshManager

    var body: some View {
        Group {
            if let s = manager.status {
                runningSection(s)
            } else if manager.isStarting {
                Label("Starting…", systemImage: "ellipsis")
                Divider()
                Button("Quit") { NSApp.terminate(nil) }
            } else {
                stoppedSection
            }
        }
        .task { manager.boot() }
    }

    func runningSection(_ s: MeshStatus) -> some View {
        Group {
            Label {
                Text(s.model_name ?? s.node_status)
            } icon: {
                Image(systemName: "circle.fill")
                    .foregroundStyle(s.node_state == "serving" ? .green :
                                    s.node_state == "loading" ? .orange : .blue)
            }

            if s.node_state == "loading" {
                Label("Loading model…", systemImage: "hourglass")
            }

            Label("\(manager.peerCount) peer\(manager.peerCount == 1 ? "" : "s"), \(manager.meshModelCount) model\(manager.meshModelCount == 1 ? "" : "s")",
                  systemImage: "network")

            if manager.isPrivateMesh {
                Label("Private mesh", systemImage: "lock")
            }

            Divider()

            if s.llama_ready {
                Button("Open Chat") { manager.openChat() }
            }

            if s.token != nil {
                Button("Copy Join Token") { manager.copyToken() }
            }

            Button("Stop") { manager.stop() }

            Divider()
            Button("Quit") { NSApp.terminate(nil) }
        }
    }

    var stoppedSection: some View {
        Group {
            Label("Not running", systemImage: "xmark.circle")
                .foregroundStyle(.secondary)

            Divider()

            Button("Start (Public Mesh)") { manager.startPublic() }

            Button("Start Private Mesh") { manager.startPrivate() }

            Button("Join with Token…") {
                JoinWindowController.show(manager: manager)
            }

            Divider()
            Button("Quit") { NSApp.terminate(nil) }
        }
    }
}
