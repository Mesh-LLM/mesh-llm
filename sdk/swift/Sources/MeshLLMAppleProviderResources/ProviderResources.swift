import Foundation

public enum MeshLLMAppleProviderResources {
    public static var appleRuntimeRoot: URL? {
        Bundle.module.url(forResource: "apple", withExtension: nil)
    }
}
