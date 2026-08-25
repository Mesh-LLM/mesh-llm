import CryptoKit
import Foundation

struct CoreAIArtifactReference: Equatable, Sendable {
  let repository: String
  let revision: String

  init?(_ value: String) {
    let parts = value.split(separator: "@", maxSplits: 1, omittingEmptySubsequences: true)
    guard parts.count == 2,
          let repositoryPart = parts.first,
          repositoryPart.split(separator: "/").count == 2,
          !repositoryPart.hasPrefix("/"),
          !repositoryPart.hasSuffix("/"),
          repositoryPart.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "/" || $0 == "-" || $0 == "." || $0 == "_" })
    else { return nil }
    guard let revisionPart = parts.dropFirst().first.map(String.init) else { return nil }
    let revisionBytes = revisionPart.utf8
    guard revisionBytes.count == 40 || revisionBytes.count == 64,
          revisionBytes.allSatisfy({ byte in
            (byte >= 48 && byte <= 57) || (byte >= 65 && byte <= 70) || (byte >= 97 && byte <= 102)
          })
    else { return nil }
    repository = String(repositoryPart)
    revision = revisionPart.lowercased()
  }

  var cacheKey: String {
    "\(repository.replacingOccurrences(of: "/", with: "--"))--\(revision)"
  }
}

private struct CoreAIPreparationManifest: Decodable {
  let schemaVersion: Int
  let artifact: Artifact

  struct Artifact: Decodable {
    let files: [ArtifactFile]
  }

  struct ArtifactFile: Decodable {
    let path: String
    let sha256: String

    enum CodingKeys: String, CodingKey {
      case path
      case sha256
    }
  }

  enum CodingKeys: String, CodingKey {
    case schemaVersion = "schema_version"
    case artifact
  }
}

struct CoreAIArtifactCache {
  private let reference: CoreAIArtifactReference
  private let cacheDirectory: URL
  private let fileManager = FileManager.default

  init(reference: CoreAIArtifactReference, cacheDirectory: URL? = nil) {
    self.reference = reference
    self.cacheDirectory = cacheDirectory
      ?? fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("mesh-llm/apple/coreai", isDirectory: true)
  }

  func materialize() async throws -> URL {
    let destination = cacheDirectory.appendingPathComponent(reference.cacheKey, isDirectory: true)
    let marker = destination.appendingPathComponent(".mesh-coreai-ready")
    if fileManager.fileExists(atPath: marker.path), isUsableBundle(at: destination) {
      return destination
    }

    try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    if fileManager.fileExists(atPath: destination.path) {
      try fileManager.removeItem(at: destination)
    }
    let staging = cacheDirectory.appendingPathComponent(
      ".\(reference.cacheKey).staging-\(UUID().uuidString)", isDirectory: true)
    try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: staging) }

    let manifestData = try await downloadData(path: "mesh-coreai-preparation.json")
    let manifest = try JSONDecoder().decode(CoreAIPreparationManifest.self, from: manifestData)
    guard manifest.schemaVersion == 1, !manifest.artifact.files.isEmpty else {
      throw CoreAIArtifactCacheError.invalidManifest
    }

    for artifactFile in manifest.artifact.files {
      guard safeRelativePath(artifactFile.path) else {
        throw CoreAIArtifactCacheError.invalidManifest
      }
      let destinationURL = staging.appendingPathComponent(artifactFile.path)
      try fileManager.createDirectory(
        at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
      try await downloadFile(path: artifactFile.path, to: destinationURL)
      guard sha256(at: destinationURL) == artifactFile.sha256.lowercased() else {
        throw CoreAIArtifactCacheError.digestMismatch(artifactFile.path)
      }
    }

    try manifestData.write(
      to: staging.appendingPathComponent("mesh-coreai-preparation.json"), options: .atomic)
    guard isUsableBundle(at: staging) else {
      throw CoreAIArtifactCacheError.invalidBundle
    }
    try Data(reference.revision.utf8).write(
      to: staging.appendingPathComponent(".mesh-coreai-ready"), options: .atomic)
    try fileManager.moveItem(at: staging, to: destination)
    return destination
  }

  private func downloadData(path: String) async throws -> Data {
    let response = try await URLSession.shared.data(from: try remoteURL(path: path))
    guard (response.1 as? HTTPURLResponse)?.statusCode == 200 else {
      throw CoreAIArtifactCacheError.downloadFailed(path)
    }
    return response.0
  }

  private func downloadFile(path: String, to destination: URL) async throws {
    let (temporaryURL, response) = try await URLSession.shared.download(
      from: try remoteURL(path: path))
    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
      throw CoreAIArtifactCacheError.downloadFailed(path)
    }
    try fileManager.moveItem(at: temporaryURL, to: destination)
  }

  private func remoteURL(path: String) throws -> URL {
    guard let encodedPath = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
          let url = URL(string: "https://huggingface.co/\(reference.repository)/resolve/\(reference.revision)/\(encodedPath)")
    else {
      throw CoreAIArtifactCacheError.downloadFailed(path)
    }
    return url
  }

  private func safeRelativePath(_ path: String) -> Bool {
    guard !path.isEmpty, !path.hasPrefix("/") else { return false }
    let components = path.split(separator: "/")
    guard !components.isEmpty, !components.contains("..") else { return false }
    return true
  }

  private func isUsableBundle(at root: URL) -> Bool {
    let metadata = root.appendingPathComponent("metadata.json")
    let artifact = (try? fileManager.contentsOfDirectory(
      at: root, includingPropertiesForKeys: nil))?.contains { url in
        url.pathExtension == "aimodel" && fileManager.fileExists(atPath: url.path)
      } ?? false
    return fileManager.fileExists(atPath: metadata.path) && artifact
  }

  private func sha256(at url: URL) -> String {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return "" }
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try? handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

enum CoreAIArtifactCacheError: Error, LocalizedError, Equatable {
  case invalidManifest
  case invalidBundle
  case downloadFailed(String)
  case digestMismatch(String)

  var errorDescription: String? {
    switch self {
    case .invalidManifest: return "Core AI artifact manifest is invalid"
    case .invalidBundle: return "Core AI artifact cache did not contain a usable bundle"
    case .downloadFailed(let path): return "Core AI artifact download failed for \(path)"
    case .digestMismatch(let path): return "Core AI artifact digest mismatch for \(path)"
    }
  }
}
