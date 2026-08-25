import Testing

@testable import MeshAppleRuntime

@Test func coreAIArtifactReferenceDefaultsToMain() {
  #expect(CoreAIArtifactReference("meshllm/example") == CoreAIArtifactReference("meshllm/example@main"))
}

@Test func coreAIArtifactReferenceKeepsPinnedRevision() {
  let reference = CoreAIArtifactReference("meshllm/qwen3-0.6b-4bit-aimodel@88064c009da71a9e5488c5db044fbfcf07703e42")
  #expect(reference?.repository == "meshllm/qwen3-0.6b-4bit-aimodel")
  #expect(reference?.revision == "88064c009da71a9e5488c5db044fbfcf07703e42")
  #expect(reference?.cacheKey == "meshllm--qwen3-0.6b-4bit-aimodel--88064c009da71a9e5488c5db044fbfcf07703e42")
}

@Test func coreAIArtifactReferenceRejectsUnsafeValues() {
  #expect(CoreAIArtifactReference("meshllm/example@../main") == nil)
  #expect(CoreAIArtifactReference("https://huggingface.co/meshllm/example") == nil)
  #expect(CoreAIArtifactReference("meshllm/example/") == nil)
}
