package ai.meshllm

import com.sun.jna.NativeLibrary
import java.io.File
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import org.junit.Test
import uniffi.mesh_ffi.currentMeshVersion

/** Genuine native fixture. Its foreground test process also needs an owned outer deadline. */
class NativeLiveTranscriptionTest {
    @Test
    fun nativeSpeechDiscoveryValidationAndCancellation() {
        assumeTrue("Set MESH_SDK_LIVE_SPEECH=1 for native speech", System.getenv("MESH_SDK_LIVE_SPEECH") == "1")
        val inputs = SpeechInputs()
        runBlocking { NativeSpeechFixture(inputs).run() }
    }
}

private class SpeechInputs {
    private val env = System.getenv()
    val source = path("MESH_SPEECH_INPUTS")
    val model = source.resolve("ggml-tiny.bin")
    val pcm = source.resolve("jfk.s16le").readBytes()
    val artifact = path("MESH_SDK_NATIVE_LIBRARY")
    val artifactHash = digest(artifact.readBytes())
    val evidence = path("MESH_SPEECH_EVIDENCE")
    val runtimeManifest = path("MESH_SDK_RUNTIME_MANIFEST")
    val runtimeManifestHash = digest(runtimeManifest.readBytes())
    val runtimeHashes = Json.parseToJsonElement(runtimeManifest.readText()).jsonObject
        .getValue("runtime").jsonObject.getValue("files").jsonObject.values.map { it.jsonPrimitive.content }.toSet()
    val ports = listOf("MESH_SDK_API_PORT", "MESH_SDK_CONSOLE_PORT", "MESH_SDK_BIND_PORT").map {
        required(it).toInt().also { port -> check(port in 1024..65535) { "Invalid $it" } }
    }

    init {
        check(pcm.size == 352_000) { "Expected exactly 11 seconds of PCM16LE" }
        check(digest(pcm) == "a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9") { "PCM digest mismatch" }
        check(digest(model.readBytes()) == "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21") { "Tiny model digest mismatch" }
        check(artifactHash == required("MESH_SDK_NATIVE_SHA256")) { "Native library digest mismatch" }
        check(ports.distinct().size == 3) { "Fixture ports must be distinct" }
        val state = path("MESH_SDK_STATE_ROOT")
        check(state.path != File(System.getProperty("user.home")).canonicalPath && state.parentFile != null) { "State root must be isolated" }
        for (key in listOf("MESH_LLM_CONFIG", "MESH_LLM_RUNTIME_ROOT", "MESH_LLM_DATA_DIR", "MESH_LLM_PLUGIN_DIR",
            "MESH_LLM_NODE_KEY_PATH", "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
            "XDG_RUNTIME_DIR", "HF_HUB_CACHE", "HF_XET_CACHE")) {
            val destination = path(key)
            check(destination.toPath().startsWith(state.toPath()) && destination != state) { "$key must stay under the isolated state root" }
            check(!destination.exists() || (destination.isDirectory && destination.listFiles()?.isEmpty() == true)) { "$key must be fresh" }
        }
        val runtimeRoot = path("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
        check(runtimeManifest.toPath().startsWith(runtimeRoot.toPath())) { "Runtime manifest must belong to the selected bundle" }
        check(runtimeManifestHash == required("MESH_SDK_RUNTIME_MANIFEST_SHA256")) { "Runtime manifest digest mismatch" }
        check(runtimeHashes.isNotEmpty()) { "Runtime manifest must declare hashed runtime files" }
        check(!evidence.exists() || (evidence.isDirectory && evidence.listFiles()?.isEmpty() == true)) { "Evidence directory must be fresh" }
        check(evidence.isDirectory || evidence.mkdirs()) { "Cannot create evidence directory" }
        // The generated UniFFI loader reads this property; a search-path fallback is insufficient.
        val override = System.getProperty("uniffi.component.mesh_ffi.libraryOverride")
        check(override != null && File(override).isAbsolute && File(override).canonicalFile == artifact) {
            "The generated loader must select the admitted absolute native library"
        }
        check(NativeLibrary.getInstance(override).file.canonicalFile == artifact) { "JNA mapped a different library" }
    }

    private fun required(key: String): String = env[key]?.takeIf { it.isNotBlank() } ?: error("Missing $key")
    private fun path(key: String): File = File(required(key)).also { check(it.isAbsolute) { "$key must be absolute" } }.canonicalFile

    fun write(name: String, value: JsonObject) = evidence.resolve(name).writeText(value.toString() + "\n")

    fun config(): JsonObject = buildJsonObject {
        put("mode", "serve")
        put("http", buildJsonObject { put("api_port", ports[0]); put("console_port", ports[1]); put("console_ui", false) })
        put("serving", buildJsonObject { put("models", JsonArray(listOf(JsonPrimitive(model.path)))) })
        put("storage", buildJsonObject { put("isolated_config", true) })
        put("network", buildJsonObject {
            put("auto_join", false); put("publish", false); put("enumerate_host", true)
            put("bind_ip", "127.0.0.1"); put("bind_port", ports[2])
            put("disable_iroh_relays", true); put("discovery_mode", "mdns")
        })
        put("startup_timeout", buildJsonObject { put("secs", 60); put("nanos", 0) })
    }

    companion object {
        fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }
}

private class SpeechEvents(private val file: File) {
    private val started = System.nanoTime()
    private val events = mutableListOf<LiveTranscriptionEvent>()

    @Synchronized
    fun record(event: LiveTranscriptionEvent, source: String) {
        events.add(event)
        val projection = buildJsonObject {
            put("type", event.type)
            put("item_id", event.itemId?.let(::JsonPrimitive) ?: JsonNull)
            put("text", event.text?.let(::JsonPrimitive) ?: JsonNull)
            put("transcript", event.transcript?.let(::JsonPrimitive) ?: JsonNull)
            put("message", event.message?.let(::JsonPrimitive) ?: JsonNull)
            put("audio_start_ms", event.audioStartMs?.let(::JsonPrimitive) ?: JsonNull)
            put("audio_end_ms", event.audioEndMs?.let(::JsonPrimitive) ?: JsonNull)
        }
        file.appendText(buildJsonObject {
            put("elapsed_ms", (System.nanoTime() - started) / 1_000_000)
            put("source", source); put("event", projection)
        }.toString() + "\n")
    }

    @Synchronized
    fun snapshot(): List<LiveTranscriptionEvent> = events.toList()
}

private class NativeSpeechFixture(private val inputs: SpeechInputs) {
    // FFI reads may block an IO thread. Timeouts cancel the waiter, then owned
    // session cleanup must unblock the independent call before the test can pass.
    private val lifetime = SupervisorJob()
    private val scope = CoroutineScope(lifetime + Dispatchers.IO)
    private val sessions = CopyOnWriteArrayList<LiveTranscription>()

    private suspend fun <T> within(milliseconds: Long, label: String, operation: suspend () -> T): T {
        val call = scope.async { operation() }
        try {
            return withTimeout(milliseconds) { call.await() }
        } catch (failure: kotlinx.coroutines.TimeoutCancellationException) {
            error("$label timed out; fallback cleanup is not a pass")
        } finally {
            if (!call.isCompleted) call.cancel()
        }
    }

    suspend fun run() {
        assertPortsFree()
        inputs.write("identity.json", buildJsonObject {
            put("pid", ProcessHandle.current().pid()); put("version", currentMeshVersion())
            put("native_library", inputs.artifact.path); put("native_sha256", inputs.artifactHash)
            put("jna_resolved_library", NativeLibrary.getInstance(inputs.artifact.path).file.canonicalPath)
            put("ports", JsonArray(inputs.ports.map(::JsonPrimitive)))
        })
        val node = Node.fromConfig(inputs.config())
        var failure: Throwable? = null
        try {
            within(75_000, "node start") { node.start() }
            check(within(5_000, "node status") { node.status().running }) { "Started node is not running" }
            val model = within(45_000, "native discovery") { discover(node) }
            recordLoadedRuntime()
            invalidBounds(node, model)
            transcribe(node, model)
            cancelCollector(node, model)
        } catch (caught: Throwable) {
            failure = caught
        } finally {
            try { cleanup(node) } catch (caught: Throwable) {
                failure?.addSuppressed(caught) ?: run { failure = caught }
            }
            inputs.write("outcome.json", buildJsonObject {
                put("passed", failure == null); put("native_calls_settled", lifetime.children.none())
            })
            scope.cancel()
        }
        failure?.let { throw it }
    }

    private suspend fun discover(node: Node): String {
        val deadline = System.nanoTime() + 45_000_000_000
        while (System.nanoTime() < deadline) {
            val page = within(10_000, "native model list") {
                node.models.list(MeshModelQuery(workloadClass = "speech_recognition", available = true))
            }
            val model = page.data.firstOrNull(::ready)
            if (model != null) {
                val exact = within(10_000, "native model get") { node.models.get(model.id) }
                check(exact.id == model.id && ready(exact)) { "Native get must preserve the ready speech identity" }
                check(exact.operations == model.operations && exact.workloadClass == model.workloadClass) { "Native list/get capability mismatch" }
                inputs.write("native-discovery.json", buildJsonObject {
                    put("id", exact.id); put("kind", exact.kind); put("workload_class", exact.workloadClass)
                    put("availability", exact.availability); put("operations", JsonArray(exact.operations.map(::JsonPrimitive)))
                    put("local_available_runtime_ready", true)
                })
                return exact.id
            }
            delay(100)
        }
        error("Native catalog never advertised a ready local speech model")
    }

    private fun ready(model: MeshModel): Boolean = model.kind == "model" && model.workloadClass == "speech_recognition" &&
        model.availability == "available" && "live_transcription" in model.operations &&
        model.providers.any { it.local && it.available && it.runtime?.ready == true }

    private fun recordLoadedRuntime() {
        val paths = if (System.getProperty("os.name").lowercase().contains("mac")) {
            val process = NativeLibrary.getProcess()
            val count = process.getFunction("_dyld_image_count").invokeInt(emptyArray<Any>())
            (0 until count).mapNotNull { index ->
                process.getFunction("_dyld_get_image_name").invokePointer(arrayOf(index))?.getString(0)
            }
        } else {
            check(System.getProperty("os.name").lowercase().contains("linux")) { "Mapped runtime verification supports macOS or Linux" }
            File("/proc/self/maps").readLines().mapNotNull { line -> line.indexOf('/').takeIf { it >= 0 }?.let { line.substring(it) } }
        }
        val images = paths.map(::File).filter { image ->
            listOf("libggml", "libllama", "libmtmd").any { image.name.startsWith(it) }
        }.map { it.canonicalFile }.distinct().sortedBy { it.path }
        check(images.any { it.name.startsWith("libllama.") }) { "No actual mapped llama runtime found" }
        val mapped = images.map { image ->
            val hash = SpeechInputs.digest(image.readBytes())
            check(hash in inputs.runtimeHashes) { "Mapped runtime image is not admitted by the manifest" }
            buildJsonObject { put("path", image.path); put("sha256", hash) }
        }
        // Also require the library selected by UniFFI to be in the actual process image list.
        check(paths.map(::File).any { it.canonicalFile == inputs.artifact }) { "The admitted FFI library is not actually mapped" }
        inputs.write("model-loaded-identity.json", buildJsonObject {
            put("runtime_manifest", inputs.runtimeManifest.path); put("runtime_manifest_sha256", inputs.runtimeManifestHash)
            put("mapped_runtime_images", JsonArray(mapped)); put("mapped_native_library", inputs.artifact.path)
            put("native_sha256", SpeechInputs.digest(inputs.artifact.readBytes()))
        })
    }

    private suspend fun start(node: Node, model: String, step: UInt = 400u): LiveTranscription = within(35_000, "session start") {
        node.startLiveTranscription(LiveTranscriptionOptions(model = model, language = "en", sampleRate = 16000u,
            turnDetection = null, stepMs = step, tentative = true)).also { sessions.add(it) }
    }

    private suspend fun invalidBounds(node: Node, model: String) {
        for (step in listOf(199u, 5001u)) {
            val rejection = runCatching { start(node, model, step) }.exceptionOrNull()
                ?: error("Invalid step $step created a session")
            val message = rejection.message.orEmpty()
            check("200" in message && "5000" in message) { "Invalid step must report the supported range" }
            inputs.write("invalid-$step.json", buildJsonObject {
                put("step_ms", step.toLong()); put("error_type", rejection.javaClass.name); put("message", message)
            })
        }
    }

    private suspend fun transcribe(node: Node, model: String) {
        val session = start(node, model)
        val events = SpeechEvents(inputs.evidence.resolve("speech.jsonl"))
        val reader = scope.async {
            while (true) { events.record(session.nextEvent() ?: break, "reader") }
        }
        try {
            val pcm = ByteBuffer.wrap(inputs.pcm).order(ByteOrder.LITTLE_ENDIAN)
            repeat(110) {
                val samples = List(1600) { pcm.short }
                within(10_000, "audio send") { session.sendAudio(samples) }
                delay(100)
            }
            check(!pcm.hasRemaining()) { "The full pinned clip must be sent exactly once" }
            val tail = within(20_000, "speech stop") { session.stop() }
            within(5_000, "reader EOF") { reader.await() }
            tail.forEach { events.record(it, "stop") }
            inputs.write("sent-audio.json", buildJsonObject {
                put("samples", 176000); put("sends", 110); put("sample_rate", 16000)
                put("step_ms", 400); put("tentative", true); put("pcm_sha256", SpeechInputs.digest(inputs.pcm))
            })
            assertTranscript(events.snapshot())
            val closed = runCatching { within(2_000, "repeat stop") { session.stop() } }.exceptionOrNull()
            check(closed?.message?.contains("closed", ignoreCase = true) == true) { "Repeated stop must report a closed session" }
            check(within(2_000, "ended read") { session.nextEvent() } == null) { "Stopped reader must be at EOF" }
        } catch (failure: Throwable) {
            runCatching { within(3_000, "failed speech close") { session.close() } }
            runCatching { within(3_000, "failed reader cleanup") { reader.await() } }
            throw failure
        }
        within(3_000, "close speech") { session.close(); session.close() }
    }

    private fun assertTranscript(events: List<LiveTranscriptionEvent>) {
        val completed = events.filter { it.type == "completed" }
        val text = completed.firstOrNull()?.transcript.orEmpty()
        inputs.write("transcript.json", buildJsonObject { put("transcript", text) })
        check(completed.size == 1) { "Expected exactly one completed utterance" }
        check(events.filter { it.type == "delta" }.joinToString("") { it.text.orEmpty() } == text) { "Delta/completed mismatch" }
        check(events.any { it.type == "tentative" && !it.text.isNullOrBlank() }) { "Missing tentative text" }
        check(events.none { it.type == "error" }) { "Native speech emitted an error" }
        check(events.indexOfFirst { it.type == "completed" } > events.indexOfLast { it.type == "delta" }) { "Completed must follow the final delta" }
        val words = Regex("[a-z0-9]+").findAll(text.lowercase()).joinToString(" ") { it.value }
        check(Regex("fellow americans.*ask not.*can do for you.*ask what you can do for your country").containsMatchIn(words)) {
            "JFK clauses missing or out of order"
        }
        for (phrase in listOf("fellow americans", "can do for you", "what you can do", "do for your country")) {
            check(Regex("\\b$phrase\\b").findAll(words).count() == 1) { "Expected one occurrence of $phrase" }
        }
    }

    private suspend fun cancelCollector(node: Node, model: String) {
        val session = start(node, model)
        val entered = CompletableDeferred<Unit>()
        val count = AtomicInteger()
        val events = SpeechEvents(inputs.evidence.resolve("cancellation-events.jsonl"))
        val collector = scope.async(start = CoroutineStart.UNDISPATCHED) {
            session.events().onStart { entered.complete(Unit) }.collect {
                count.incrementAndGet(); events.record(it, "collector")
            }
        }
        try {
            withTimeout(2_000) { entered.await() }
            delay(200)
            check(collector.isActive && !collector.isCompleted && count.get() == 0) {
                "Empty-session collector must be started and pending before cancellation"
            }
            collector.cancel()
            withTimeout(3_000) { collector.join() }
            val ended = runCatching { collector.await() }.exceptionOrNull()
            check(ended is CancellationException) { "Collector failed instead of being cancelled" }
            check(collector.isCancelled) { "Collector did not terminate by cancellation" }
            check(count.get() == 0) { "Empty session emitted an event during cancellation" }
            check(within(3_000, "cancellation native EOF") { session.nextEvent() } == null) {
                "Cancellation must close the native reader before explicit cleanup"
            }
            inputs.write("cancellation.json", buildJsonObject {
                put("collector_started", true); put("pending_after_ms", 200)
                put("collector_terminated", true); put("native_eof_before_explicit_close", true)
            })
        } catch (failure: Throwable) {
            collector.cancel()
            runCatching { within(3_000, "fallback cancellation close") { session.close() } }
            runCatching { withTimeout(3_000) { collector.join() } }
            throw failure
        }
        within(3_000, "repeat cancellation close") { session.close(); session.close() }
        check(within(2_000, "repeat cancelled read") { session.nextEvent() } == null) { "Cancelled reader must stay at EOF" }
    }

    private suspend fun cleanup(node: Node) {
        val failures = mutableListOf<Throwable>()
        for (session in sessions) {
            runCatching { within(5_000, "cleanup session") { session.close() } }.exceptionOrNull()?.let(failures::add)
        }
        runCatching { within(30_000, "node stop") { node.stop() } }.exceptionOrNull()?.let(failures::add)
        runCatching {
            check(!within(5_000, "stopped status") { node.status().running }) { "Node remains running after stop" }
            withTimeout(5_000) { while (lifetime.children.any()) delay(10) }
            assertPortsFree()
        }.exceptionOrNull()?.let(failures::add)
        inputs.write("cleanup.json", buildJsonObject {
            put("passed", failures.isEmpty()); put("owned_ports", JsonArray(inputs.ports.map(::JsonPrimitive)))
        })
        if (failures.isNotEmpty()) {
            val first = failures.first()
            failures.drop(1).forEach(first::addSuppressed)
            throw first
        }
    }

    private fun assertPortsFree() {
        for (port in inputs.ports.take(2)) {
            ServerSocket().use { socket -> socket.reuseAddress = true; socket.bind(InetSocketAddress("127.0.0.1", port)) }
        }
        DatagramSocket(null).use { socket ->
            socket.reuseAddress = false
            socket.bind(InetSocketAddress("127.0.0.1", inputs.ports[2]))
        }
    }
}
