use super::predicate::token;

mod autoincrement;

#[derive(Debug, Eq, PartialEq)]
pub(super) enum Collation {
    Binary,
    Named(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Deferrability {
    NotDeferrable,
    Deferrable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InitialTiming {
    Immediate,
    Deferred,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct Column {
    pub(super) collation: Collation,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct ForeignKey {
    pub(super) columns: Vec<String>,
    pub(super) deferrability: Deferrability,
    pub(super) initial_timing: InitialTiming,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct Table {
    pub(super) columns: Vec<Column>,
    pub(super) foreign_keys: Vec<ForeignKey>,
    pub(super) autoincrement_column: Option<String>,
}

pub(super) fn parse(sql: &str) -> Option<Table> {
    let clauses = top_level_clauses(sql)?;
    let mut columns = Vec::new();
    let mut foreign_keys = Vec::new();
    let mut autoincrement_column = None;
    for clause in clauses {
        let tokens = token::tokenize(clause);
        let first = tokens.first()?.as_str();
        if is_table_constraint(first) {
            if autoincrement::contains_keyword(&tokens) {
                return None;
            }
            if let Some(foreign_key) = table_foreign_key(&tokens)? {
                foreign_keys.push(foreign_key);
            }
            continue;
        }

        let column_name = tokens.first()?.clone();
        if let autoincrement::Clause::Column(column) = autoincrement::parse(&tokens)?
            && autoincrement_column.replace(column).is_some()
        {
            return None;
        }
        columns.push(Column {
            collation: column_collation(&tokens)?,
        });
        if top_level_position(&tokens, "REFERENCES").is_some() {
            foreign_keys.push(foreign_key(&tokens, vec![column_name])?);
        }
    }
    Some(Table {
        columns,
        foreign_keys,
        autoincrement_column,
    })
}

fn is_table_constraint(first: &str) -> bool {
    matches!(
        first,
        "CONSTRAINT" | "PRIMARY" | "UNIQUE" | "CHECK" | "FOREIGN"
    )
}

fn column_collation(tokens: &[String]) -> Option<Collation> {
    let Some(position) = top_level_position(tokens, "COLLATE") else {
        return Some(Collation::Binary);
    };
    match tokens.get(position + 1)?.as_str() {
        "BINARY" => Some(Collation::Binary),
        name => Some(Collation::Named(name.to_owned())),
    }
}

fn table_foreign_key(tokens: &[String]) -> Option<Option<ForeignKey>> {
    let Some(foreign) = top_level_position(tokens, "FOREIGN") else {
        return Some(None);
    };
    if tokens.get(foreign + 1).map(String::as_str) != Some("KEY")
        || tokens.get(foreign + 2).map(String::as_str) != Some("(")
    {
        return None;
    }
    let close = tokens[foreign + 3..]
        .iter()
        .position(|value| value == ")")?
        + foreign
        + 3;
    let columns = tokens[foreign + 3..close].to_vec();
    if columns.is_empty() {
        return None;
    }
    foreign_key(tokens, columns).map(Some)
}

fn foreign_key(tokens: &[String], columns: Vec<String>) -> Option<ForeignKey> {
    let references = top_level_position(tokens, "REFERENCES")?;
    let (deferrability, initial_timing) = foreign_key_timing(&tokens[references + 1..])?;
    Some(ForeignKey {
        columns,
        deferrability,
        initial_timing,
    })
}

fn foreign_key_timing(tokens: &[String]) -> Option<(Deferrability, InitialTiming)> {
    let mut deferrability = Deferrability::NotDeferrable;
    let mut initial_timing = InitialTiming::Immediate;
    let mut depth = 0_u32;
    for (position, value) in tokens.iter().enumerate() {
        match value.as_str() {
            "(" => depth += 1,
            ")" => depth = depth.checked_sub(1)?,
            "DEFERRABLE" if depth == 0 => {
                deferrability = if position > 0 && tokens[position - 1] == "NOT" {
                    Deferrability::NotDeferrable
                } else {
                    Deferrability::Deferrable
                };
            }
            "INITIALLY" if depth == 0 => {
                initial_timing = match tokens.get(position + 1)?.as_str() {
                    "IMMEDIATE" => InitialTiming::Immediate,
                    "DEFERRED" => InitialTiming::Deferred,
                    _ => return None,
                };
            }
            _ => {}
        }
    }
    Some((deferrability, initial_timing))
}

fn top_level_position(tokens: &[String], expected: &str) -> Option<usize> {
    let mut depth = 0_u32;
    for (position, value) in tokens.iter().enumerate() {
        match value.as_str() {
            "(" => depth += 1,
            ")" => depth = depth.checked_sub(1)?,
            _ if depth == 0 && value == expected => return Some(position),
            _ => {}
        }
    }
    None
}

fn top_level_clauses(sql: &str) -> Option<Vec<&str>> {
    let bytes = sql.as_bytes();
    let body_start = bytes.iter().position(|value| *value == b'(')? + 1;
    let mut clauses = Vec::new();
    let mut clause_start = body_start;
    let mut depth = 1_u32;
    let mut quote = None;
    let mut position = body_start;
    while position < bytes.len() {
        if let Some(closing) = quote {
            if bytes[position] == closing {
                if closing != b']' && bytes.get(position + 1) == Some(&closing) {
                    position += 2;
                    continue;
                }
                quote = None;
            }
            position += 1;
            continue;
        }
        match bytes[position] {
            b'\'' | b'"' | b'`' => quote = Some(bytes[position]),
            b'[' => quote = Some(b']'),
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    clauses.push(sql[clause_start..position].trim());
                    return Some(clauses);
                }
            }
            b',' if depth == 1 => {
                clauses.push(sql[clause_start..position].trim());
                clause_start = position + 1;
            }
            _ => {}
        }
        position += 1;
    }
    None
}

#[cfg(test)]
mod tests;
