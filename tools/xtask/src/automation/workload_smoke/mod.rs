pub(crate) mod comparison;
mod embedding;
mod encoding;
mod http;
pub(crate) mod media_comparison;
mod responses;
pub(crate) mod tts_oracle;
use crate::command::DynResult;
use serde_json::json;
use std::{fs, io::Write, path::Path};

const INPUTS: [&str; 3] = [
    "search_query: distributed GPU inference",
    "search_document: GPUs share one language model over a mesh",
    "search_document: A recipe for tomato soup",
];

const RERANK_QUERY: &str = "distributed GPU inference";
const RERANK_DOCUMENTS: [&str; 2] = [
    "GPUs share one language model over a mesh",
    "A recipe for tomato soup",
];
const ENCODER_DECODER_PROMPT: &str = "translate English to German: The house is wonderful.";

#[derive(Clone, Copy)]
enum Workload {
    Embedding,
    Rerank,
    EncoderDecoder,
    Ocr,
    SpeechSynthesis,
    SpeechRecognition,
}
impl Workload {
    fn parse(text: &str) -> DynResult<Self> {
        match text {
            "embedding" => Ok(Self::Embedding),
            "rerank" => Ok(Self::Rerank),
            "encoder_decoder" => Ok(Self::EncoderDecoder),
            "ocr" => Ok(Self::Ocr),
            "speech_synthesis" => Ok(Self::SpeechSynthesis),
            "speech_recognition" => Ok(Self::SpeechRecognition),
            _ => Err("unsupported workload class".into()),
        }
    }
}

fn request(base: &str, path: &str, payload: &serde_json::Value) -> DynResult<Vec<u8>> {
    http::post(
        &format!("{base}{path}"),
        "application/json",
        serde_json::to_vec(payload)?,
        if path == "/audio/speech" {
            "audio/wav"
        } else {
            "application/json"
        },
    )
}

fn execute(base: &str, model: &str, workload: Workload, media: Option<&Path>) -> DynResult<()> {
    match workload {
        Workload::Embedding => {
            let numeric = request(
                base,
                "/embeddings",
                &json!({"model":model,"input":INPUTS,"encoding_format":"float"}),
            )?;
            let encoded = request(
                base,
                "/embeddings",
                &json!({"model":model,"input":INPUTS[0],"encoding_format":"base64"}),
            )?;
            embedding::validate(&numeric, &encoded, model)
        }
        Workload::Rerank => responses::rerank(&request(
            base,
            "/rerank",
            &json!({"model":model,"query":RERANK_QUERY,"documents":RERANK_DOCUMENTS,"return_documents":true}),
        )?),
        Workload::EncoderDecoder => responses::completion(&request(
            base,
            "/completions",
            &json!({"model":model,"prompt":ENCODER_DECODER_PROMPT,"max_tokens":32,"temperature":0}),
        )?),
        Workload::Ocr => {
            let media = media.ok_or("OCR requires --media-path")?;
            let image = encoding::encode(&fs::read(media)?);
            responses::ocr(&request(
                base,
                "/chat/completions",
                &json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"Read all visible text. Return only the transcription."},{"type":"image_url","image_url":{"url":format!("data:image/png;base64,{image}")}}]}],"max_tokens":32,"temperature":0}),
            )?)
        }
        Workload::SpeechSynthesis => responses::audio(&request(
            base,
            "/audio/speech",
            &json!({"model":model,"input":"The mesh is ready.","voice":"default","response_format":"wav"}),
        )?),
        Workload::SpeechRecognition => {
            let media = media.ok_or("speech recognition requires --media-path")?;
            let filename = media
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("invalid media filename")?;
            if model.contains(['\r', '\n']) || filename.contains(['\r', '\n', '"']) {
                return Err("invalid multipart field".into());
            }
            let boundary = "mesh-llm-workload-smoke";
            let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n").into_bytes();
            body.extend(fs::read(media)?);
            body.extend(format!("\r\n--{boundary}--\r\n").bytes());
            responses::transcription(&http::post(
                &format!("{base}/audio/transcriptions"),
                &format!("multipart/form-data; boundary={boundary}"),
                body,
                "application/json",
            )?)
        }
    }
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let mut values = std::collections::BTreeMap::new();
    if !args.len().is_multiple_of(2) {
        return Err("workload-smoke requires named option values".into());
    }
    for pair in args.as_chunks::<2>().0 {
        if !matches!(
            pair[0].as_str(),
            "--base-url" | "--model" | "--class" | "--media-path"
        ) || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("unknown or duplicate workload option".into());
        }
    }
    let base = *values.get("--base-url").ok_or("missing --base-url")?;
    let model = *values.get("--model").ok_or("missing --model")?;
    let class = *values.get("--class").ok_or("missing --class")?;
    execute(
        base.trim_end_matches('/'),
        model,
        Workload::parse(class)?,
        values
            .get("--media-path")
            .filter(|path| !path.is_empty())
            .map(Path::new),
    )?;
    writeln!(
        crate::cli_output::stdout(),
        "OpenAI HTTP {class} smoke passed: model={model}"
    )?;
    Ok(())
}
