//! Canonical generation-11 stage admission descriptors.

/// Current descriptor schema carried by stage-control generation 11.
pub const STAGE_ADMISSION_DESCRIPTOR_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageAdmissionDescriptor {
    pub version: u32,
    pub package_id: String,
    pub plan_id: String,
    pub execution_contract: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub resident_tensor_ids: Vec<String>,
    pub sidecars: Vec<StageAdmissionSidecar>,
    pub profiles: Vec<StageAdmissionProfile>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct StageAdmissionSidecar {
    pub kind: StageAdmissionSidecarKind,
    pub artifact_id: String,
    pub name: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum StageAdmissionSidecarKind {
    Mmproj,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageAdmissionProfile {
    pub profile_id: String,
    pub graph_identity: String,
    pub profile_identity: String,
    pub slice_identity: String,
    pub source_snapshot_identity: String,
    pub graph_configuration_id: String,
    pub backend_id: String,
    pub activation_imports: Vec<String>,
    pub activation_exports: Vec<String>,
    pub activation_import_bindings: Vec<String>,
    pub activation_export_bindings: Vec<String>,
}

impl From<StageAdmissionDescriptor> for crate::proto::stage::StageAdmissionDescriptor {
    fn from(descriptor: StageAdmissionDescriptor) -> Self {
        Self {
            version: descriptor.version,
            package_id: descriptor.package_id,
            plan_id: descriptor.plan_id,
            execution_contract: descriptor.execution_contract,
            layer_start: descriptor.layer_start,
            layer_end: descriptor.layer_end,
            resident_tensor_ids: descriptor.resident_tensor_ids,
            sidecars: descriptor.sidecars.into_iter().map(Into::into).collect(),
            profiles: descriptor.profiles.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<crate::proto::stage::StageAdmissionDescriptor> for StageAdmissionDescriptor {
    type Error = crate::StageFrameError;

    fn try_from(
        descriptor: crate::proto::stage::StageAdmissionDescriptor,
    ) -> Result<Self, Self::Error> {
        crate::validate_stage_admission_descriptor(&descriptor)?;
        Ok(Self {
            version: descriptor.version,
            package_id: descriptor.package_id,
            plan_id: descriptor.plan_id,
            execution_contract: descriptor.execution_contract,
            layer_start: descriptor.layer_start,
            layer_end: descriptor.layer_end,
            resident_tensor_ids: descriptor.resident_tensor_ids,
            sidecars: descriptor
                .sidecars
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
            profiles: descriptor.profiles.into_iter().map(Into::into).collect(),
        })
    }
}

impl From<StageAdmissionSidecar> for crate::proto::stage::StageAdmissionSidecar {
    fn from(sidecar: StageAdmissionSidecar) -> Self {
        Self {
            kind: match sidecar.kind {
                StageAdmissionSidecarKind::Mmproj => {
                    crate::proto::stage::StageAdmissionSidecarKind::Mmproj as i32
                }
            },
            artifact_id: sidecar.artifact_id,
            name: sidecar.name,
        }
    }
}

impl TryFrom<crate::proto::stage::StageAdmissionSidecar> for StageAdmissionSidecar {
    type Error = crate::StageFrameError;

    fn try_from(sidecar: crate::proto::stage::StageAdmissionSidecar) -> Result<Self, Self::Error> {
        let kind = match crate::proto::stage::StageAdmissionSidecarKind::try_from(sidecar.kind) {
            Ok(crate::proto::stage::StageAdmissionSidecarKind::Mmproj) => {
                StageAdmissionSidecarKind::Mmproj
            }
            _ => {
                return Err(crate::StageFrameError::InvalidStageAdmissionDescriptor(
                    "unsupported sidecar kind",
                ));
            }
        };
        Ok(Self {
            kind,
            artifact_id: sidecar.artifact_id,
            name: sidecar.name,
        })
    }
}

impl From<StageAdmissionProfile> for crate::proto::stage::StageAdmissionProfile {
    fn from(profile: StageAdmissionProfile) -> Self {
        Self {
            profile_id: profile.profile_id,
            graph_identity: profile.graph_identity,
            profile_identity: profile.profile_identity,
            slice_identity: profile.slice_identity,
            source_snapshot_identity: profile.source_snapshot_identity,
            graph_configuration_id: profile.graph_configuration_id,
            backend_id: profile.backend_id,
            activation_imports: profile.activation_imports,
            activation_exports: profile.activation_exports,
            activation_import_bindings: profile.activation_import_bindings,
            activation_export_bindings: profile.activation_export_bindings,
        }
    }
}

impl From<crate::proto::stage::StageAdmissionProfile> for StageAdmissionProfile {
    fn from(profile: crate::proto::stage::StageAdmissionProfile) -> Self {
        Self {
            profile_id: profile.profile_id,
            graph_identity: profile.graph_identity,
            profile_identity: profile.profile_identity,
            slice_identity: profile.slice_identity,
            source_snapshot_identity: profile.source_snapshot_identity,
            graph_configuration_id: profile.graph_configuration_id,
            backend_id: profile.backend_id,
            activation_imports: profile.activation_imports,
            activation_exports: profile.activation_exports,
            activation_import_bindings: profile.activation_import_bindings,
            activation_export_bindings: profile.activation_export_bindings,
        }
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;

    #[test]
    fn execution_contract_survives_control_protocol_roundtrip() {
        let descriptor = super::StageAdmissionDescriptor {
            version: super::STAGE_ADMISSION_DESCRIPTOR_VERSION,
            package_id: format!("sha256:{}", "a".repeat(64)),
            plan_id: format!("skippy-plan:v1:{}", "b".repeat(64)),
            execution_contract: r#"{"version":1,"decoder":["weight"]}"#.into(),
            layer_start: 0,
            layer_end: 1,
            resident_tensor_ids: vec!["weight".into()],
            sidecars: vec![],
            profiles: vec![super::StageAdmissionProfile {
                profile_id: "decode".into(),
                graph_identity: "graph".into(),
                profile_identity: "profile".into(),
                slice_identity: "slice".into(),
                source_snapshot_identity: "snapshot".into(),
                graph_configuration_id: "config".into(),
                backend_id: "cpu".into(),
                activation_imports: vec![],
                activation_exports: vec![],
                activation_import_bindings: vec![],
                activation_export_bindings: vec![],
            }],
        };
        let wire: crate::proto::stage::StageAdmissionDescriptor = descriptor.clone().into();
        let bytes = wire.encode_to_vec();
        let decoded =
            crate::proto::stage::StageAdmissionDescriptor::decode(bytes.as_slice()).unwrap();
        let restored = super::StageAdmissionDescriptor::try_from(decoded).unwrap();
        assert_eq!(restored, descriptor);
    }
}
