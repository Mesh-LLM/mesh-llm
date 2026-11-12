use super::*;
use image::DynamicImage;
use serde_json::json;

fn fixture_uri(format: ImageFormat) -> String {
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::new_rgb8(2, 2)
        .write_to(&mut encoded, format)
        .unwrap();
    let mime = match format {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::WebP => "image/webp",
        _ => unreachable!(),
    };
    format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(encoded.into_inner())
    )
}

fn chat(text: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"choices":[{"message":{"content":text}}]})).unwrap()
}

fn event(value: Value) -> String {
    format!("data: {value}\n\n")
}

fn chat_delta(index: u64, text: &str) -> String {
    event(json!({"choices":[{"index":index,"delta":{"content":text}}]}))
}

fn response_delta(text: &str) -> String {
    event(
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg1","delta":text}),
    )
}

fn large_opaque_noise_png() -> (String, Vec<u8>) {
    let mut seed = 0x1234_5678_u32;
    let pixels = image::RgbImage::from_fn(384, 384, |_, _| {
        let mut rgb = [0_u8; 3];
        for channel in &mut rgb {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *channel = seed as u8;
        }
        image::Rgb(rgb)
    });
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(pixels)
        .write_to(&mut png, ImageFormat::Png)
        .unwrap();
    let bytes = png.into_inner();
    assert!(
        bytes.len() > 400_000,
        "fixture must exercise a large raster"
    );
    let uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    );
    assert!(
        uri.len() > 500_000,
        "fixture must exercise a long Markdown URL"
    );
    (uri, bytes)
}

#[test]
fn accepts_backend_chat_with_large_opaque_noise_png_markdown() {
    let (uri, bytes) = large_opaque_noise_png();
    let body = serde_json::to_vec(&json!({"choices":[{
        "index":0,
        "message":{"role":"assistant","content":format!("Here is the generated image.\n\n![Generated image]({uri})")},
        "finish_reason":"stop"
    }]})).unwrap();
    let images = images_from_response("application/json", &body)
        .unwrap_or_else(|error| panic!("canonical large backend PNG rejected: {error:#}"));
    assert_eq!(images.len(), 1);
    assert!(images[0].bytes == bytes, "encoded raster must be preserved");
    assert!(images[0].data_uri == uri, "data URI must be preserved");
}

fn named_response_event(value: Value) -> String {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    format!("event: {kind}\n{}", event(value))
}

fn large_response_stream(text: &str) -> String {
    use skippy_inference_api as api;
    let mut sequence = 0_i32;
    let mut next = || {
        let current = sequence;
        sequence += 1;
        current
    };
    let mut frames = vec![
        api::responses_stream_created_event_with_sequence("noise", 1, next()),
        api::responses_stream_output_item_added_event("msg1", next()),
        api::responses_stream_content_part_added_event("msg1", next()),
    ];
    frames.extend(text.as_bytes().chunks(8192).map(|chunk| {
        api::responses_stream_delta_event_with_logprobs_and_sequence(
            "msg1",
            std::str::from_utf8(chunk).unwrap(),
            None,
            next(),
        )
    }));
    frames.extend([
        api::responses_stream_text_done_event_with_sequence("msg1", text, next()),
        api::responses_stream_content_part_done_event("msg1", text, next()),
        api::responses_stream_output_item_done_event("msg1", text, next()),
        api::responses_stream_completed_event_with_sequence(
            "resp1",
            1,
            "noise",
            "msg1",
            text,
            None,
            next(),
        ),
    ]);
    let mut body = frames
        .into_iter()
        .map(named_response_event)
        .collect::<String>();
    body.push_str("event: done\ndata: [DONE]\n\n");
    body
}

#[test]
fn accepts_large_noise_image_in_stable_responses_sse_deltas_and_snapshots() {
    let (uri, bytes) = large_opaque_noise_png();
    let text = format!("Here is the generated image.\n\n![Generated image]({uri})");
    let body = large_response_stream(&text);
    let images = images_from_response("text/event-stream", body.as_bytes())
        .unwrap_or_else(|error| panic!("canonical large backend SSE rejected: {error:#}"));
    assert_eq!(images.len(), 1);
    assert!(images[0].bytes == bytes, "encoded raster must be preserved");
    assert!(images[0].data_uri == uri, "data URI must be preserved");
}

#[test]
fn rejects_duplicate_json_keys_that_could_hide_an_image_from_screening() {
    let image =
        serde_json::to_string(&format!("![result]({})", fixture_uri(ImageFormat::Png))).unwrap();
    let bodies = [
        [
            r#"{"choices":[{"message":{"content":"#,
            image.as_str(),
            r#", "content":"plain"}}]}"#,
        ]
        .concat(),
        [
            r#"{"choices":[{"message":{"content":"#,
            image.as_str(),
            r#", "cont\u0065nt":"plain"}}]}"#,
        ]
        .concat(),
        [
            r#"{"choices":[{"message":{"content":"#,
            image.as_str(),
            r#"}}],"choices":[{"message":{"content":"plain"}}]}"#,
        ]
        .concat(),
    ];
    for body in bodies {
        // The ordinary value parser would inspect only the last field.
        let value = serde_json::from_str(&body).unwrap();
        assert_eq!(texts_from_json(&value).unwrap(), vec!["plain"]);
        assert!(images_from_response("application/json", body.as_bytes()).is_err());
    }
}

#[test]
fn rejects_duplicate_keys_in_chat_and_responses_sse_before_releasing_raw_frames() {
    let image =
        serde_json::to_string(&format!("![result]({})", fixture_uri(ImageFormat::Png))).unwrap();
    let chat = [
        r#"data: {"choices":[{"index":0,"delta":{"content":"#,
        image.as_str(),
        r#", "content":"plain"}}]}"#,
        "\n\ndata: [DONE]\n\n",
    ]
    .concat();
    let responses = [
        r#"data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg1","delta":"#,
        image.as_str(),
        r#", "delt\u0061":"plain"}"#,
        "\n\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n",
    ]
    .concat();
    for body in [chat, responses] {
        assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
    }
}

#[test]
fn fully_decodes_each_supported_raster_format() {
    for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
        let uri = fixture_uri(format);
        let images = images_from_response(
            "application/json; charset=utf-8",
            &chat(&format!("Here it is. ![result]({uri})")),
        )
        .unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].data_uri, uri);
        assert_eq!(image::guess_format(&images[0].bytes).unwrap(), format);
    }
}

#[test]
fn rejects_transparency_that_could_change_between_classifier_and_browser() {
    for alpha in [0, 127, 255] {
        let image = image::RgbaImage::from_pixel(2, 2, image::Rgba([20, 40, 60, alpha]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(encoded.into_inner())
        );
        assert_eq!(
            images_from_response("application/json", &chat(&format!("![result]({uri})"))).is_ok(),
            alpha == 255
        );
    }
}

#[test]
fn inspects_all_choices_parts_and_reasoning_but_deduplicates_pixels() {
    let png = fixture_uri(ImageFormat::Png);
    let jpeg = fixture_uri(ImageFormat::Jpeg);
    let webp = fixture_uri(ImageFormat::WebP);
    let body = serde_json::to_vec(&json!({"choices":[
        {"message":{"content":[{"type":"text","text":format!("![a]({png})")},{"type":"text","text":format!("![b]({jpeg})")}]}},
        {"message":{"content":format!("![duplicate]({png})"),"reasoning_content":format!("![reasoning]({webp})")}}
    ]})).unwrap();
    assert_eq!(
        images_from_response("application/json", &body)
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn reconstructs_chat_sse_image_split_across_base64_and_markdown() {
    let text = format!("![result]({})", fixture_uri(ImageFormat::Png));
    let body = format!(
        "{}{}{}data: [DONE]\n\n",
        chat_delta(0, &text[..9]),
        chat_delta(0, &text[9..41]),
        chat_delta(0, &text[41..])
    );
    assert_eq!(
        images_from_response("text/event-stream", body.as_bytes())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn reconstructs_responses_sse_and_inspects_terminal_snapshots() {
    let text = format!("![result]({})", fixture_uri(ImageFormat::Png));
    let item = json!({"type":"message","content":[{"type":"output_text","text":text}]});
    let body = format!(
        "{}{}{}{}{}",
        response_delta(&text[..25]),
        response_delta(&text[25..]),
        event(json!({"type":"response.output_text.done","text":text})),
        event(json!({"type":"response.output_item.done","item":item})),
        event(json!({"type":"response.completed","response":{"output":[item]}}))
    );
    assert_eq!(
        images_from_response("text/event-stream", body.as_bytes())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn examines_image_present_only_in_completed_response() {
    let text = format!("![result]({})", fixture_uri(ImageFormat::Png));
    let body = event(
        json!({"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":text}]}]}}),
    );
    assert_eq!(
        images_from_response("text/event-stream", body.as_bytes())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn responses_json_collects_output_and_ignores_arbitrary_metadata() {
    let uri = fixture_uri(ImageFormat::Png);
    let body = serde_json::to_vec(&json!({"object":"response","metadata":{"unrendered":format!("![x]({uri})")},"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]})).unwrap();
    assert!(
        images_from_response("application/json", &body)
            .unwrap()
            .is_empty()
    );
    let body = serde_json::to_vec(&json!({"object":"response","output":[{"type":"message","content":[{"type":"output_text","text":format!("![x]({uri})")}]}]})).unwrap();
    assert_eq!(
        images_from_response("application/json", &body)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn joins_text_parts_and_chronological_ui_deltas_before_image_inspection() {
    let text = format!("![result]({})", fixture_uri(ImageFormat::Png));
    let split = text.find("image/").unwrap() + 2;
    let body = serde_json::to_vec(&json!({"choices":[{"message":{"content":[
        {"type":"text","text":text[..split]}, {"type":"text","text":text[split..]}
    ]}}]}))
    .unwrap();
    assert_eq!(
        images_from_response("application/json", &body)
            .unwrap()
            .len(),
        1
    );
    let first = json!({"type":"response.output_text.delta","output_index":0,"item_id":"one","delta":text[..split]});
    let second = json!({"type":"response.output_text.delta","output_index":1,"item_id":"two","delta":text[split..]});
    let body = format!("{}{}data: [DONE]\n\n", event(first), event(second));
    assert_eq!(
        images_from_response("text/event-stream", body.as_bytes())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn completed_responses_json_with_null_error_still_screens_its_output() {
    let uri = fixture_uri(ImageFormat::Png);
    for (text, expected_images) in [
        ("hello there".to_owned(), Some(0)),
        (format!("Here it is. ![result]({uri})"), Some(1)),
        ("![result](data:image/png;base64,broken)".to_owned(), None),
    ] {
        let body =
            skippy_inference_api::translate_chat_completion_to_responses(&chat(&text)).unwrap();
        let mut value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "response");
        assert_eq!(value["status"], "completed");
        assert_eq!(value.get("error"), Some(&Value::Null));
        let result = images_from_response("application/json", &body);
        if let Some(count) = expected_images {
            let images = result.unwrap();
            assert_eq!(images.len(), count);
            if count == 1 {
                assert_eq!(images[0].data_uri, uri);
            }
        } else {
            assert!(
                result.is_err(),
                "null error must not bypass artifact screening"
            );
        }

        value["error"] = json!({"message":"ordinary error"});
        assert!(
            images_from_response("application/json", &serde_json::to_vec(&value).unwrap()).is_err(),
            "a real error must not mix with model output"
        );
    }
}

#[test]
fn error_envelopes_cannot_hide_image_artifacts_or_mix_with_output() {
    let image = format!("![result]({})", fixture_uri(ImageFormat::Png));
    for value in [
        json!({"error":{"message":image}}),
        json!({"error":{"message":"ordinary error"},"choices":[{"message":{"content":image}}]}),
        json!({"error":{"message":"ordinary error"},"output_text":image}),
        json!({"error":{"message":"![x](https://example.invalid/image.png)"}}),
    ] {
        assert!(
            images_from_response("application/json", &serde_json::to_vec(&value).unwrap()).is_err()
        );
    }
    assert!(
        images_from_response(
            "application/json",
            br#"{"error":{"message":"not available"}}"#
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn rejects_sse_error_envelopes_before_choices() {
    let uri = fixture_uri(ImageFormat::Png);
    let body = format!(
        "{}data: [DONE]\n\n",
        event(json!({"choices":[],"error":{"message":format!("![result]({uri})")}}))
    );
    let error = images_from_response("text/event-stream", body.as_bytes())
        .expect_err("an error frame must not hide an image behind empty choices");
    assert_eq!(
        error.to_string(),
        "image response did not complete successfully"
    );

    let body = format!(
        "{}data: [DONE]\n\n",
        event(json!({"choices":[{"index":0,"delta":{"content":"hello"}}],"error":null}))
    );
    assert!(
        images_from_response("text/event-stream", body.as_bytes())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn plain_text_and_regular_text_completions_are_allowed() {
    assert!(
        images_from_response("text/plain", b"hello there")
            .unwrap()
            .is_empty()
    );
    assert!(
        images_from_response("application/json", &chat("hello there"))
            .unwrap()
            .is_empty()
    );
    assert!(
        images_from_response("application/json", br#"{"choices":[{"text":"hello"}]}"#)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rejects_sse_without_complete_terminal_frame_or_after_failure() {
    let delta = chat_delta(0, "hello");
    for body in [
        delta.clone(),
        format!("{delta}data: [DONE]\n"),
        event(json!({"type":"response.incomplete"})),
        event(json!({"type":"response.failed"})),
    ] {
        assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
    }
    let body = format!("data: [DONE]\n\n{delta}");
    assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
}

#[test]
fn choices_cannot_join_to_manufacture_a_complete_image() {
    let text = format!("![result]({})", fixture_uri(ImageFormat::Png));
    let split = text.len() / 2;
    let body = format!(
        "{}{}data: [DONE]\n\n",
        chat_delta(0, &text[..split]),
        chat_delta(1, &text[split..])
    );
    assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
}

#[test]
fn malformed_image_cannot_hide_in_a_final_snapshot() {
    let body = format!(
        "{}{}",
        response_delta("hello"),
        event(
            json!({"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"![x](data:image/png;base64,broken)"}]}]}})
        )
    );
    assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
}

#[test]
fn rejects_unsupported_external_and_noncanonical_image_references() {
    let uri = fixture_uri(ImageFormat::Png);
    for text in [
        "![x](https://example.invalid/image.png)".to_owned(),
        "![x](mesh://blob/image)".to_owned(),
        "![x](data:image/gif;base64,R0lGODlh)".to_owned(),
        "![x](data:image/svg+xml;base64,PHN2Zz4=)".to_owned(),
        format!("![x][ref]\n\n[ref]: {uri}"),
        format!("<img src=\"{uri}\">"),
        "<svg><image href=\"https://example.invalid/a.png\" /></svg>".to_owned(),
        uri.clone(),
        format!("`![x]({uri})`"),
        format!("![x]({uri}"),
    ] {
        assert!(
            images_from_response("application/json", &chat(&text)).is_err(),
            "unsupported image form was admitted"
        );
    }
}

#[test]
fn supports_inline_angle_destination_and_title() {
    let uri = fixture_uri(ImageFormat::Png);
    let text = format!("![generated](<{uri}> \"result\")");
    assert_eq!(
        images_from_response("text/plain", text.as_bytes())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn rejects_structured_images_until_the_protocol_admits_them() {
    for body in [
        json!({"choices":[{"message":{"content":[{"type":"image_url","image_url":{"url":"https://example.invalid/x"}}]}}]}),
        json!({"object":"response","output":[{"type":"image_generation_call","result":"unexamined"}]}),
        json!({"choices":[{"message":{"content":"hello","images":[{"image_url":{"url":"https://example.invalid/x"}}]}}]}),
        json!({"choices":[{"delta":{"content":"unadmitted JSON transport"}}]}),
    ] {
        assert!(
            images_from_response("application/json", &serde_json::to_vec(&body).unwrap()).is_err()
        );
    }
    for image_event in [
        json!({"type":"response.image_generation_call.partial_image","partial_image_b64":"unexamined"}),
        json!({"type":"response.output_item.added","item":{"type":"image_generation_call","result":"unexamined"}}),
    ] {
        let body = format!("{}data: [DONE]\n\n", event(image_event));
        assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
    }
}

#[test]
fn rejects_structured_images_in_chat_stream_deltas() {
    let uri = fixture_uri(ImageFormat::Png);
    for (field, image) in [
        ("image", json!(uri)),
        ("image_url", json!({"url":uri})),
        ("images", json!([{"image_url":{"url":uri}}])),
    ] {
        let mut delta = json!({"content":""});
        delta[field] = image;
        let body = format!(
            "{}data: [DONE]\n\n",
            event(json!({"choices":[{"index":0,"delta":delta}]}))
        );
        let error = images_from_response("text/event-stream", body.as_bytes())
            .expect_err("structured image deltas must not pass as image-free text");
        assert_eq!(
            error.to_string(),
            "structured image output is not supported by image screening",
            "streaming field {field} must fail closed"
        );
    }
}

#[test]
fn rejects_invalid_base64_mime_mismatch_and_incomplete_pixels() {
    let uri = fixture_uri(ImageFormat::Png);
    let truncated = uri.split_once(',').unwrap();
    let mut bytes = base64::engine::general_purpose::STANDARD
        .decode(truncated.1)
        .unwrap();
    bytes.truncate(bytes.len() / 2);
    for bad_uri in [
        "data:image/png;base64,invalid!".to_owned(),
        "data:image/png;base64,".to_owned(),
        uri.replacen("image/png", "image/jpeg", 1),
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        ),
    ] {
        assert!(decode_image(&bad_uri).is_err());
    }
}

#[test]
fn bounds_response_uri_dimensions_and_reference_count() {
    assert!(images_from_response("text/plain", &vec![b'x'; MAX_RESPONSE_BYTES + 1]).is_err());
    assert!(decode_image(&"x".repeat(MAX_IMAGE_URI_CHARS + 1)).is_err());
    for dimensions in [(0, 1), (1, 0), (MAX_IMAGE_DIMENSION + 1, 1), (8192, 8192)] {
        assert!(validate_dimensions(dimensions).is_err());
    }
    assert!(validate_dimensions((4096, 4096)).is_ok());
    let text = format!("![x]({}) ", fixture_uri(ImageFormat::Png)).repeat(MAX_IMAGE_REFERENCES + 1);
    assert!(images_from_response("text/plain", text.as_bytes()).is_err());
}

#[test]
fn bounds_repeated_sse_image_snapshot_inspection() {
    let text = format!("![x]({})", fixture_uri(ImageFormat::Png));
    let snapshot = event(json!({"type":"response.output_text.done","text":text}));
    let body = format!(
        "{}{}",
        snapshot.repeat(MAX_IMAGE_INSPECTIONS + 1),
        event(json!({"type":"response.completed","response":{}}))
    );
    assert!(images_from_response("text/event-stream", body.as_bytes()).is_err());
}

#[test]
fn rejects_png_animation_and_trailing_images() {
    let uri = fixture_uri(ImageFormat::Png);
    let bytes = decode_image(&uri).unwrap().bytes;
    let mut animated = bytes.clone();
    let iend = animated.len() - 12;
    // acTL is rejected before the image decoder, independently of its CRC.
    animated.splice(
        iend..iend,
        [
            0, 0, 0, 8, b'a', b'c', b'T', b'L', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(validate_png_chunks(&animated).is_err());
    let mut appended = bytes.clone();
    appended.extend_from_slice(&bytes);
    assert!(validate_png_chunks(&appended).is_err());
}

#[test]
fn rejects_invalid_png_checksum_even_after_last_pixel_chunk() {
    let mut bytes = decode_image(&fixture_uri(ImageFormat::Png)).unwrap().bytes;
    // Pixel-only decode need not consume IEND, so the container walk validates
    // its CRC as well as the chunks required to produce the displayed raster.
    *bytes.last_mut().unwrap() ^= 1;
    let uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    );
    assert!(decode_image(&uri).is_err());
}

#[test]
fn rejects_animated_webp_and_inconsistent_container_length() {
    let uri = fixture_uri(ImageFormat::WebP);
    let mut bytes = decode_image(&uri).unwrap().bytes;
    bytes.extend_from_slice(b"ANIM\x00\x00\x00\x00");
    let size = u32::try_from(bytes.len() - 8).unwrap();
    bytes[4..8].copy_from_slice(&size.to_le_bytes());
    assert!(validate_webp_chunks(&bytes).is_err());
    bytes.truncate(bytes.len() - 8);
    assert!(validate_webp_chunks(&bytes).is_err());
}

#[test]
fn rejects_multipicture_and_concatenated_jpeg() {
    let uri = fixture_uri(ImageFormat::Jpeg);
    let bytes = decode_image(&uri).unwrap().bytes;
    let mut multiple = bytes.clone();
    multiple.extend_from_slice(&bytes);
    assert!(validate_jpeg_segments(&multiple).is_err());
    let mut mpo = bytes;
    mpo.splice(2..2, [0xff, 0xe2, 0, 6, b'M', b'P', b'F', 0]);
    assert!(validate_jpeg_segments(&mpo).is_err());
}

#[test]
fn image_debug_does_not_leak_data_uri() {
    let image = decode_image(&fixture_uri(ImageFormat::Png)).unwrap();
    let debug = format!("{image:?}");
    assert!(!debug.contains("data:"));
    assert!(!debug.contains(&image.data_uri));
}
