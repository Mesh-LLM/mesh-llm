//! Finite local HTTP responder: not a model and not frontend qualification.
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
pub(super) struct Server {
    pub url: String,
    pub calls: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
}
impl Server {
    pub fn new(mode: &str, mutation: &str) -> Self {
        Self::with_listener(TcpListener::bind("127.0.0.1:0").unwrap(), mode, mutation)
    }
    pub fn with_listener(listener: TcpListener, mode: &str, mutation: &str) -> Self {
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let mode = mode.to_owned();
        let mutation = mutation.to_owned();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(12);
            while !stopping.load(Ordering::SeqCst) && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let Some((method, request)) = request(&mut stream) else {
                            continue;
                        };
                        let index = {
                            let mut calls = observed.lock().unwrap();
                            calls.push(request.clone());
                            calls.len()
                        };
                        if mutation == "stall" {
                            while !stopping.load(Ordering::SeqCst) && Instant::now() < deadline {
                                thread::sleep(Duration::from_millis(10));
                            }
                            continue;
                        }
                        let (status, mut response) = if mode == "contract" {
                            refusal(&method, &request)
                        } else {
                            (200, answer(&request, index, &mutation))
                        };
                        let status = mutate_contract(status, &mut response, &request, &mutation);
                        let bytes = if mutation == "malformed" {
                            b"{".to_vec()
                        } else if mutation == "oversized" {
                            vec![b'x'; 1048577]
                        } else {
                            serde_json::to_vec(&response).unwrap()
                        };
                        let header = format!(
                            "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n",
                            bytes.len() + usize::from(mutation == "incomplete")
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(&bytes);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            url,
            calls,
            stop,
            worker: Mutex::new(Some(worker)),
        }
    }
}
impl Server {
    pub fn finish(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            assert!(worker.join().is_ok(), "System One fixture worker failed");
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(worker) = self.worker.get_mut()
            && let Some(worker) = worker.take()
        {
            let _ = worker.join();
        }
    }
}
fn request(stream: &mut TcpStream) -> Option<(String, Value)> {
    // BSD/macOS accept inherits O_NONBLOCK from the listener. Timed finite
    // reads require blocking mode; an initial WouldBlock is not a refusal.
    stream.set_nonblocking(false).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(1))).ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(1)))
        .ok()?;
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..n]);
        if bytes.len() > 65536 {
            return None;
        }
        if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end]).ok()?;
    let start = header.lines().next()?;
    let method = start.split_whitespace().next()?.to_owned();
    assert!(start.contains(" /systemone "));
    let length = header
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|n| n.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    if length > 65536 {
        return None;
    }
    while bytes.len() < header_end + length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..n]);
    }
    Some((
        method,
        serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap_or(Value::Null),
    ))
}
fn refusal(method: &str, request: &Value) -> (u16, Value) {
    let (status, kind, code, message) =
        if method == "GET" {
            (
                405,
                "invalid_request_error",
                "method_not_allowed",
                "method not allowed",
            )
        } else if request.is_null()
            || request["model"] == "definitely-not-loaded"
            || request["questions"]
                .as_object()
                .is_some_and(|q| q.is_empty())
        {
            (
                400,
                "invalid_request_error",
                "invalid_value",
                "invalid request",
            )
        } else if ["sequential", "steps", "samples", "think", "images"]
            .iter()
            .any(|key| request.get(*key).is_some())
        {
            (
                400,
                "invalid_request_error",
                "unsupported_model_feature",
                "unsupported",
            )
        } else if request["questions"].as_object().unwrap().values().any(|q| {
            match q["type"].as_str() {
                Some("choice") => !(2..=26).contains(&q["criteria"].as_object().unwrap().len()),
                Some("score") => !(2..=10).contains(&q["criteria"].as_array().unwrap().len()),
                _ => false,
            }
        }) {
            (
                400,
                "invalid_request_error",
                "invalid_value",
                "invalid options",
            )
        } else {
            (
                502,
                "server_error",
                "service_unavailable",
                "System One requires DiffusionGemma",
            )
        };
    (
        status,
        json!({"error":{"type":kind,"code":code,"message":message}}),
    )
}
fn answer(request: &Value, index: usize, mutation: &str) -> Value {
    let other = request["state"].as_str().unwrap().contains("segmentation");
    let high = if other && mutation != "constant" {
        0.2
    } else if index == 8 && mutation == "leaked" {
        0.6
    } else {
        0.8
    };
    let mut answers = serde_json::Map::new();
    for (key, q) in request["questions"].as_object().unwrap() {
        let mut value = match q["type"].as_str().unwrap() {
            "noul" => json!({"type":"noul","noul":high}),
            "choice" => {
                let options = q["criteria"].as_object().unwrap();
                let selected_index = if high >= (1.0 - high) / (options.len() - 1) as f64 {
                    0
                } else {
                    1
                };
                let first = options.keys().nth(selected_index).unwrap();
                let probs: serde_json::Map<_, _> = options
                    .keys()
                    .enumerate()
                    .map(|(i, key)| {
                        (
                            key.clone(),
                            json!(if i == 0 {
                                high
                            } else {
                                (1.0 - high) / (options.len() - 1) as f64
                            }),
                        )
                    })
                    .collect();
                json!({"type":"choice","choice":first,"probabilities":probs,"confidence":0.2})
            }
            "score" => {
                let count = q["criteria"].as_array().unwrap().len();
                let probs: serde_json::Map<_, _> = (0..count)
                    .map(|i| (i.to_string(), json!(1.0 / count as f64)))
                    .collect();
                let legend: serde_json::Map<_, _> = (0..count)
                    .map(|i| (i.to_string(), json!(format!("level {i}"))))
                    .collect();
                json!({"type":"score","score":(count-1) as f64/2.0,"probabilities":probs,"legend":legend,"confidence":0.0})
            }
            _ => unreachable!(),
        };
        if mutation == "unnormalized" && value["type"] == "choice" {
            value["probabilities"]["team_0"] = json!(0.99);
        }
        if mutation == "boolean" {
            if value["type"] == "noul" {
                value["noul"] = json!(true);
            } else {
                value["confidence"] = json!(true);
            }
        }
        if mutation == "bad-score" && value["type"] == "score" {
            value["score"] = json!(0.0);
        }
        if mutation == "not-argmax" && value["type"] == "choice" {
            value["choice"] = json!("team_1");
        }
        if mutation == "bad-choice" && value["type"] == "choice" {
            value["choice"] = json!("outside");
        }
        if mutation == "repeat-invalid" && index == 8 {
            value["noul"] = json!(2.0);
        }
        answers.insert(key.clone(), value);
    }
    let mut body = json!({"model":request["model"],"usage":{"input_tokens":4,"output_tokens":0},"answers":answers});
    match mutation {
        "alias" if index == 5 => body["model"] = json!("canonical"),
        "usage" => body["usage"] = Value::Null,
        "generated" => body["usage"]["output_tokens"] = json!(1),
        _ => (),
    };
    body
}

fn mutate_contract(status: u16, response: &mut Value, request: &Value, mutation: &str) -> u16 {
    if mutation == "wrong-code" {
        response["error"]["code"] = json!("wrong");
    }
    if mutation == "unsupported-success" && response["error"]["code"] == "unsupported_model_feature"
    {
        return 200;
    }
    if mutation == "unsupported-code" && response["error"]["code"] == "unsupported_model_feature" {
        response["error"]["code"] = json!("service_unavailable");
    }
    if mutation == "arch-success" && status == 502 {
        return 200;
    }
    if status == 502 {
        match mutation {
            "arch-message" => response["error"]["message"] = json!("generic backend failure"),
            "arch-envelope" => response["error"] = json!("untyped refusal"),
            _ => (),
        };
    }
    let count = |key: &str, array: bool| -> Option<usize> {
        let criteria = &request["questions"][key]["criteria"];
        if array {
            criteria.as_array().map(Vec::len)
        } else {
            criteria.as_object().map(serde_json::Map::len)
        }
    };
    let accepted = match mutation {
        "choice-min" => count("team", false) == Some(1),
        "choice-max" => count("team", false) == Some(27),
        "score-min" => count("urgency", true) == Some(1),
        "score-max" => count("urgency", true) == Some(11),
        _ => false,
    };
    if accepted {
        502
    } else if mutation == "missing-refusal" {
        200
    } else if mutation == "redirect" {
        302
    } else {
        status
    }
}
