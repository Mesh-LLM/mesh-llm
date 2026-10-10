use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt;
use std::marker::PhantomData;

#[derive(Debug, Default)]
pub(super) enum Field<T> {
    #[default]
    Missing,
    Null,
    Present(T),
    Invalid(FieldError),
}

#[derive(Debug)]
pub(super) enum FieldError {
    Expected(&'static str),
    Duplicate,
    UnknownVerdict(String),
}

impl fmt::Display for FieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expected(expected) => write!(formatter, "expected {expected}"),
            Self::Duplicate => formatter.write_str("duplicate consumed member"),
            Self::UnknownVerdict(verdict) => write!(formatter, "unknown verdict {verdict:?}"),
        }
    }
}

impl<T: ReportValue> Field<T> {
    pub(super) fn value(&self) -> Option<&T> {
        match self {
            Self::Present(value) => Some(value),
            Self::Missing | Self::Null | Self::Invalid(_) => None,
        }
    }

    pub(super) fn errors(&self, path: &str, nullable: bool, failures: &mut Vec<String>) {
        match self {
            Self::Present(value) => value.errors(path, failures),
            Self::Invalid(error) => failures.push(format!("{path}: {error}")),
            Self::Null if !nullable => failures.push(format!("{path}: expected {}", T::EXPECTED)),
            Self::Missing | Self::Null => {}
        }
    }

    pub(super) fn read_member<'de, A: MapAccess<'de>>(
        &mut self,
        map: &mut A,
    ) -> Result<(), A::Error> {
        if matches!(self, Self::Missing) {
            *self = map.next_value()?;
        } else {
            map.next_value::<IgnoredAny>()?;
            *self = Self::Invalid(FieldError::Duplicate);
        }
        Ok(())
    }
}

pub(super) trait ReportValue: Sized {
    const EXPECTED: &'static str;

    fn string(_value: String) -> Result<Self, FieldError> {
        Err(FieldError::Expected(Self::EXPECTED))
    }
    fn unsigned(_value: u64) -> Result<Self, FieldError> {
        Err(FieldError::Expected(Self::EXPECTED))
    }
    fn boolean(_value: bool) -> Result<Self, FieldError> {
        Err(FieldError::Expected(Self::EXPECTED))
    }
    fn map<'de, A: MapAccess<'de>>(mut map: A) -> Result<Field<Self>, A::Error> {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(Field::Invalid(FieldError::Expected(Self::EXPECTED)))
    }
    fn sequence<'de, A: SeqAccess<'de>>(mut sequence: A) -> Result<Field<Self>, A::Error> {
        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(Field::Invalid(FieldError::Expected(Self::EXPECTED)))
    }
    fn errors(&self, _path: &str, _failures: &mut Vec<String>) {}
}

impl<'de, T: ReportValue> Deserialize<'de> for Field<T> {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        decoder.deserialize_any(FieldVisitor(PhantomData))
    }
}

struct FieldVisitor<T>(PhantomData<T>);

impl<'de, T: ReportValue> Visitor<'de> for FieldVisitor<T> {
    type Value = Field<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(T::EXPECTED)
    }
    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Field::Null)
    }
    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(candidate(T::boolean(value)))
    }
    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(candidate(T::unsigned(value)))
    }
    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(candidate(
            u64::try_from(value)
                .map_err(|_| FieldError::Expected(T::EXPECTED))
                .and_then(T::unsigned),
        ))
    }
    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(Field::Invalid(FieldError::Expected(T::EXPECTED)))
    }
    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(candidate(T::string(value.to_owned())))
    }
    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(candidate(T::string(value)))
    }
    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        T::map(map)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, sequence: A) -> Result<Self::Value, A::Error> {
        T::sequence(sequence)
    }
}

fn candidate<T>(value: Result<T, FieldError>) -> Field<T> {
    match value {
        Ok(value) => Field::Present(value),
        Err(error) => Field::Invalid(error),
    }
}

impl ReportValue for String {
    const EXPECTED: &'static str = "string";
    fn string(value: String) -> Result<Self, FieldError> {
        Ok(value)
    }
}

impl ReportValue for bool {
    const EXPECTED: &'static str = "boolean";
    fn boolean(value: bool) -> Result<Self, FieldError> {
        Ok(value)
    }
}

impl<T: ReportValue> ReportValue for Vec<Field<T>> {
    const EXPECTED: &'static str = "array";
    fn sequence<'de, A: SeqAccess<'de>>(mut sequence: A) -> Result<Field<Self>, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = sequence.next_element()? {
            items.push(item);
        }
        Ok(Field::Present(items))
    }
    fn errors(&self, path: &str, failures: &mut Vec<String>) {
        for (index, item) in self.iter().enumerate() {
            item.errors(&format!("{path}[{index}]"), false, failures);
        }
    }
}

macro_rules! string_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub(super) struct $name(pub(super) String);
        impl ReportValue for $name {
            const EXPECTED: &'static str = "string";
            fn string(value: String) -> Result<Self, FieldError> {
                Ok(Self(value))
            }
        }
    };
}
string_type!(BuilderFile);
string_type!(ConstructorName);

#[derive(PartialEq, Eq, Hash)]
pub(super) struct BuilderKey(pub(super) BuilderFile, pub(super) ConstructorName);

#[derive(Debug)]
pub(super) struct SchemaVersion(pub(super) u64);
impl ReportValue for SchemaVersion {
    const EXPECTED: &'static str = "unsigned integer schema version";
    fn unsigned(value: u64) -> Result<Self, FieldError> {
        Ok(Self(value))
    }
}

macro_rules! bounded_integer {
    ($name:ident) => {
        #[derive(Debug)]
        pub(super) struct $name(u64);
        impl ReportValue for $name {
            const EXPECTED: &'static str = "integer in 0..=9223372036854775807";
            fn unsigned(value: u64) -> Result<Self, FieldError> {
                if i64::try_from(value).is_ok() {
                    Ok(Self(value))
                } else {
                    Err(FieldError::Expected(Self::EXPECTED))
                }
            }
        }
    };
}
bounded_integer!(ReportCount);
bounded_integer!(ReportOffset);

impl ReportCount {
    pub(super) fn exact(&self) -> u128 {
        u128::from(self.0)
    }
}

impl ReportValue for [ReportOffset; 2] {
    const EXPECTED: &'static str = "two integer offsets in 0..=9223372036854775807";
    fn sequence<'de, A: SeqAccess<'de>>(mut sequence: A) -> Result<Field<Self>, A::Error> {
        let first = sequence.next_element::<Field<ReportOffset>>()?;
        let second = sequence.next_element::<Field<ReportOffset>>()?;
        let mut extra = false;
        while sequence.next_element::<IgnoredAny>()?.is_some() {
            extra = true;
        }
        Ok(match (first, second, extra) {
            (Some(Field::Present(first)), Some(Field::Present(second)), false) => {
                Field::Present([first, second])
            }
            _ => Field::Invalid(FieldError::Expected(Self::EXPECTED)),
        })
    }
    fn errors(&self, _path: &str, _failures: &mut Vec<String>) {
        let [ReportOffset(_start), ReportOffset(_end)] = self;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Verdict {
    Transformable,
    AlreadyTransformed,
    SupportedAuxiliary,
    SupportedWholeModel,
    UnsupportedShape,
    Error,
}
impl ReportValue for Verdict {
    const EXPECTED: &'static str = "verdict string";
    fn string(value: String) -> Result<Self, FieldError> {
        match value.as_str() {
            "transformable" => Ok(Self::Transformable),
            "already_transformed" => Ok(Self::AlreadyTransformed),
            "supported_auxiliary" => Ok(Self::SupportedAuxiliary),
            "supported_whole_model" => Ok(Self::SupportedWholeModel),
            "unsupported_shape" => Ok(Self::UnsupportedShape),
            "error" => Ok(Self::Error),
            _ => Err(FieldError::UnknownVerdict(value)),
        }
    }
}

macro_rules! record {
    ($name:ident { $($field:ident: $kind:ty),* $(,)? }) => {
        #[derive(Debug, Default)]
        pub(super) struct $name { $(pub(super) $field: Field<$kind>),* }
        impl ReportValue for $name {
            const EXPECTED: &'static str = "object";
            fn map<'de, A: MapAccess<'de>>(mut map: A) -> Result<Field<Self>, A::Error> {
                let mut record = Self::default();
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        $(key if key == stringify!($field).trim_start_matches("r#") => record.$field.read_member(&mut map)?,)*
                        _ => { map.next_value::<IgnoredAny>()?; }
                    }
                }
                Ok(Field::Present(record))
            }
            fn errors(&self, path: &str, failures: &mut Vec<String>) {
                $(self.$field.errors(&format!("{path}.{}", stringify!($field).trim_start_matches("r#")), false, failures);)*
            }
        }
    };
}

record!(ValidateReport {
    schema_version: SchemaVersion, llama_cpp_commit: String, generator_version: String,
    builders: Vec<Field<ValidateBuilder>>, summary: SummaryCounts,
});
record!(IdempotenceReport { builders: Vec<Field<IdempotenceBuilder>> });
record!(IdempotenceBuilder {
    file: BuilderFile,
    verdict: String
});
record!(ValidateBuilder {
    file: BuilderFile, constructor: ConstructorName, verdict: Verdict, proof: ProofFields,
    edits: Vec<Field<EditFields>>, unsupported_reason: String,
});
record!(SummaryCounts {
    transformable: ReportCount,
    already_transformed: ReportCount,
    supported_auxiliary: ReportCount,
    supported_whole_model: ReportCount,
    unsupported_shape: ReportCount,
    error: ReportCount,
});
record!(ProofFields {
    r#loop: LoopFields, activation_in: String, activation_out: String,
    embedding_owner: bool, output_owner: bool, terminal_predicates: Vec<Field<String>>,
    nonlocal_exits: Vec<Field<String>>, execution_scope: String, scope_evidence: Vec<Field<String>>,
});
record!(LoopFields {
    var: String,
    start: String,
    end: String
});
record!(EditFields {
    kind: String,
    file: String,
    text: String,
    text_ref: String,
    range: [ReportOffset; 2],
});

pub(super) enum Report {
    Validate(Box<ValidateReport>),
    Idempotence(IdempotenceReport),
}
