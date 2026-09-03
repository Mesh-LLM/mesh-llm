use crate::{
    DiagnosticEventKind, EventSystemHealthEventKind, FactData, GenerationEventKind,
    KvRuntimeStateEventKind, ModelAvailabilityEventKind, ModelLoadingEventKind,
    ModelPreparationEventKind, ModelUnloadingEventKind, NativeRuntimeEventKind,
    NodeAvailabilityEventKind, PrefillEventKind, RequestEventKind, ResourceHealthEventKind,
    SessionEventKind, StageTopologyEventKind,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FamilyFact<K> {
    kind: K,
    data: FactData,
}

impl<K> FamilyFact<K> {
    #[must_use]
    pub fn new(kind: K) -> Self {
        Self {
            kind,
            data: FactData::default(),
        }
    }

    #[must_use]
    pub const fn with_data(kind: K, data: FactData) -> Self {
        Self { kind, data }
    }

    #[must_use]
    pub const fn kind(&self) -> &K {
        &self.kind
    }

    #[must_use]
    pub const fn data(&self) -> &FactData {
        &self.data
    }
}

pub type NativeRuntimeFact = FamilyFact<NativeRuntimeEventKind>;
pub type ModelPreparationFact = FamilyFact<ModelPreparationEventKind>;
pub type ModelLoadingFact = FamilyFact<ModelLoadingEventKind>;
pub type ModelAvailabilityFact = FamilyFact<ModelAvailabilityEventKind>;
pub type ModelUnloadingFact = FamilyFact<ModelUnloadingEventKind>;
pub type StageTopologyFact = FamilyFact<StageTopologyEventKind>;
pub type SessionFact = FamilyFact<SessionEventKind>;
pub type RequestFact = FamilyFact<RequestEventKind>;
pub type PrefillFact = FamilyFact<PrefillEventKind>;
pub type GenerationFact = FamilyFact<GenerationEventKind>;
pub type KvRuntimeStateFact = FamilyFact<KvRuntimeStateEventKind>;
pub type ResourceHealthFact = FamilyFact<ResourceHealthEventKind>;
pub type DiagnosticFact = FamilyFact<DiagnosticEventKind>;
pub type NodeAvailabilityFact = FamilyFact<NodeAvailabilityEventKind>;
pub type EventSystemHealthFact = FamilyFact<EventSystemHealthEventKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeFact {
    NativeRuntime(NativeRuntimeFact),
    ModelPreparation(ModelPreparationFact),
    ModelLoading(ModelLoadingFact),
    ModelAvailability(ModelAvailabilityFact),
    ModelUnloading(ModelUnloadingFact),
    StageTopology(StageTopologyFact),
    Session(SessionFact),
    Request(RequestFact),
    Prefill(PrefillFact),
    Generation(GenerationFact),
    KvRuntimeState(KvRuntimeStateFact),
    ResourceHealth(ResourceHealthFact),
    Diagnostic(DiagnosticFact),
    NodeAvailability(NodeAvailabilityFact),
    EventSystemHealth(EventSystemHealthFact),
}

impl RuntimeFact {
    #[must_use]
    pub const fn data(&self) -> &FactData {
        match self {
            Self::NativeRuntime(fact) => fact.data(),
            Self::ModelPreparation(fact) => fact.data(),
            Self::ModelLoading(fact) => fact.data(),
            Self::ModelAvailability(fact) => fact.data(),
            Self::ModelUnloading(fact) => fact.data(),
            Self::StageTopology(fact) => fact.data(),
            Self::Session(fact) => fact.data(),
            Self::Request(fact) => fact.data(),
            Self::Prefill(fact) => fact.data(),
            Self::Generation(fact) => fact.data(),
            Self::KvRuntimeState(fact) => fact.data(),
            Self::ResourceHealth(fact) => fact.data(),
            Self::Diagnostic(fact) => fact.data(),
            Self::NodeAvailability(fact) => fact.data(),
            Self::EventSystemHealth(fact) => fact.data(),
        }
    }

    #[must_use]
    pub const fn kind_id(&self) -> &'static str {
        match self {
            Self::NativeRuntime(fact) => fact.kind().as_str(),
            Self::ModelPreparation(fact) => fact.kind().as_str(),
            Self::ModelLoading(fact) => fact.kind().as_str(),
            Self::ModelAvailability(fact) => fact.kind().as_str(),
            Self::ModelUnloading(fact) => fact.kind().as_str(),
            Self::StageTopology(fact) => fact.kind().as_str(),
            Self::Session(fact) => fact.kind().as_str(),
            Self::Request(fact) => fact.kind().as_str(),
            Self::Prefill(fact) => fact.kind().as_str(),
            Self::Generation(fact) => fact.kind().as_str(),
            Self::KvRuntimeState(fact) => fact.kind().as_str(),
            Self::ResourceHealth(fact) => fact.kind().as_str(),
            Self::Diagnostic(fact) => fact.kind().as_str(),
            Self::NodeAvailability(fact) => fact.kind().as_str(),
            Self::EventSystemHealth(fact) => fact.kind().as_str(),
        }
    }
}
