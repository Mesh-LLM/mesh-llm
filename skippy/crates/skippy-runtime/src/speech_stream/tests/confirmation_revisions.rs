use super::*;

#[test]
fn a_revised_prior_phrase_and_leading_comma_do_not_replay_the_last_word() {
    // Captured base.en/400 ms passes91–95. The preceding output omitted the
    // third utterance's "ask not", although the latest hypothesis restored it.
    let canonical = "and so my fellow Americans what you country can do for you ask what you can do for your country and so my fellow Americans";
    let observed = "and so my fellow Americans asked not what your country can do for you ask what you can do for your country and so my fellow Americans";
    let words = |text: &str| {
        text.split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let mut stream = SpeechStream::new(manual());
    stream.confirm(words(canonical), &mut Vec::new());
    assert_eq!(stream.confirmed_prefix.align(&words(observed)), Some(27));
    assert_eq!(stream.confirmed_in_buffer, 25);
    let mut script = Script::new(vec![
        vec![(
            30_000,
            " and so my fellow Americans... asked not! what your country can do for you. ask what you can do for your country. and so my fellow Americans... asked...",
        )],
        vec![(
            30_000,
            " , ask what you can do for your country. ask what you can do for your country. and so my fellow Americans... asked me what you can do for your country.",
        )],
        vec![(
            30_000,
            " ask not! what you country can do for you. ask what you can do for your country. and so my fellow Americans. ask not!",
        )],
        vec![(
            30_000,
            " , ask not! what you country can do for you. ask what you can do for your country. and so my fellow Americans... asked not!",
        )],
        vec![(
            30_000,
            " , ask not! what you country can do for you. ask what you can do for your country. and so my fellow Americans... asked not!",
        )],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut events = Vec::new();
    for _ in 0..5 {
        events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
    }
    events.extend(stream.commit(&mut transcribe).unwrap());
    assert_eq!(deltas(&events).concat(), " asked not!");
}
