use super::create_table::{Collation, Deferrability, ForeignKey, InitialTiming, Table};

pub(super) struct TableContract {
    pub(super) name: &'static str,
    column_collation: Collation,
    foreign_keys: &'static [ForeignKeyContract],
    autoincrement_column: Option<&'static str>,
}

struct ForeignKeyContract {
    columns: &'static [&'static str],
    deferrability: Deferrability,
    initial_timing: InitialTiming,
}

impl TableContract {
    pub(super) fn matches(&self, table: &Table) -> bool {
        table
            .columns
            .iter()
            .all(|column| column.collation == self.column_collation)
            && table.foreign_keys.len() == self.foreign_keys.len()
            && table
                .foreign_keys
                .iter()
                .zip(self.foreign_keys)
                .all(|(actual, expected)| expected.matches(actual))
            && table.autoincrement_column.as_deref() == self.autoincrement_column
    }
}

impl ForeignKeyContract {
    fn matches(&self, foreign_key: &ForeignKey) -> bool {
        foreign_key
            .columns
            .iter()
            .map(String::as_str)
            .eq(self.columns.iter().copied())
            && foreign_key.deferrability == self.deferrability
            && foreign_key.initial_timing == self.initial_timing
    }
}

macro_rules! foreign_key {
    ($column:literal) => {
        ForeignKeyContract {
            columns: &[$column],
            deferrability: Deferrability::NotDeferrable,
            initial_timing: InitialTiming::Immediate,
        }
    };
}

macro_rules! table {
    ($name:literal, $autoincrement_column:expr, $foreign_keys:expr) => {
        TableContract {
            name: $name,
            column_collation: Collation::Binary,
            foreign_keys: $foreign_keys,
            autoincrement_column: $autoincrement_column,
        }
    };
}

pub(super) const TABLES: &[TableContract] = &[
    table!("artifact_pointers", None, &[foreign_key!("REQUEST_ID")]),
    table!(
        "audit_entries",
        Some("SEQUENCE"),
        &[foreign_key!("REQUEST_ID")]
    ),
    table!("cleanup_runs", None, &[]),
    table!("lifecycle_events", None, &[foreign_key!("REQUEST_ID")]),
    table!(
        "maintenance_operation_targets",
        None,
        &[foreign_key!("OPERATION_ID")]
    ),
    table!("maintenance_operations", None, &[]),
    table!("pending_artifact_deletions", None, &[]),
    table!("proxy_records", None, &[foreign_key!("REQUEST_ID")]),
    table!("summaries", None, &[]),
    table!("webhook_deliveries", None, &[foreign_key!("REQUEST_ID")]),
];
