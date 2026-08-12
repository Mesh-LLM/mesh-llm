import Foundation
import MeshLLM

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count == 3 else {
    FileHandle.standardError.write(
        Data("usage: AppleSystemHost <provider-root> <ready-file> <stop-file>\n".utf8)
    )
    exit(2)
}

let providerRoot = URL(fileURLWithPath: arguments[0], isDirectory: true)
let readyFile = URL(fileURLWithPath: arguments[1])
let stopFile = URL(fileURLWithPath: arguments[2])

Task {
    do {
        let host = try await ProviderHost.start(
            ProviderRuntimeOptions(bundleRoots: [providerRoot])
        )
        let ready = try JSONSerialization.data(withJSONObject: [
            "carrier": "swift",
            "apiBaseUrl": host.apiBaseURL.absoluteString,
        ])
        try ready.write(to: readyFile, options: .atomic)
        while !FileManager.default.fileExists(atPath: stopFile.path) {
            try await Task.sleep(for: .milliseconds(100))
        }
        try await host.stop()
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
}

RunLoop.main.run()
