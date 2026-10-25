use crate::process::Failure;

const MAX_NAME_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemberId {
    name: [u8; MAX_NAME_BYTES],
    length: usize,
    generation: u32,
}

impl MemberId {
    #[expect(
        non_upper_case_globals,
        reason = "existing retained callers use these identity names"
    )]
    pub const Seed: Self = Self::literal(b"seed");
    #[expect(
        non_upper_case_globals,
        reason = "existing retained callers use these identity names"
    )]
    pub const WorkerOne: Self = Self::literal(b"worker-one");
    #[expect(
        non_upper_case_globals,
        reason = "existing retained callers use these identity names"
    )]
    pub const WorkerTwo: Self = Self::literal(b"worker-two");

    const fn literal(bytes: &[u8]) -> Self {
        let mut name = [0; MAX_NAME_BYTES];
        let mut index = 0;
        while index < bytes.len() {
            name[index] = bytes[index];
            index += 1;
        }
        Self {
            name,
            length: bytes.len(),
            generation: 0,
        }
    }

    pub fn new(name: &str, generation: u32) -> Result<Self, Failure> {
        if name.is_empty()
            || name.len() > MAX_NAME_BYTES
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(Failure::InvalidMemberName);
        }
        Ok(Self {
            generation,
            ..Self::literal(name.as_bytes())
        })
    }

    pub fn name(&self) -> &[u8] {
        &self.name[..self.length]
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }

    pub fn next_generation(self) -> Result<Self, Failure> {
        Ok(Self {
            generation: self
                .generation
                .checked_add(1)
                .ok_or(Failure::MemberGenerationExhausted)?,
            ..self
        })
    }
}
