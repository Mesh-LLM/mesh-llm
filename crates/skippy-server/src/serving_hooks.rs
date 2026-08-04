use std::{fmt, sync::Arc};

use anyhow::Result;

use crate::{GenerationReceiptConfig, LinearProposalIngressConfig, TokenizerCapability};

/// Constructs product-neutral serving hooks after Skippy has loaded the model
/// and can expose its authoritative tokenizer capability.
pub trait ModelServingHooksFactory: Send + Sync {
    fn create(&self, tokenizer: TokenizerCapability) -> Result<ModelServingHooks>;
}

impl fmt::Debug for dyn ModelServingHooksFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ModelServingHooksFactory")
    }
}

pub type SharedModelServingHooksFactory = Arc<dyn ModelServingHooksFactory>;

/// Optional lifecycle observer and proposal source for local serving.
#[derive(Clone, Default)]
pub struct ModelServingHooks {
    generation_receipt: Option<GenerationReceiptConfig>,
    linear_proposal_ingress: Option<LinearProposalIngressConfig>,
}

impl ModelServingHooks {
    #[must_use]
    pub fn with_generation_receipt(mut self, config: GenerationReceiptConfig) -> Self {
        self.generation_receipt = Some(config);
        self
    }

    #[must_use]
    pub fn with_linear_proposal_ingress(mut self, config: LinearProposalIngressConfig) -> Self {
        self.linear_proposal_ingress = Some(config);
        self
    }

    #[must_use]
    pub fn new(
        generation_receipt: GenerationReceiptConfig,
        linear_proposal_ingress: LinearProposalIngressConfig,
    ) -> Self {
        Self::default()
            .with_generation_receipt(generation_receipt)
            .with_linear_proposal_ingress(linear_proposal_ingress)
    }

    pub fn generation_receipt(&self) -> Option<GenerationReceiptConfig> {
        self.generation_receipt.clone()
    }

    pub fn linear_proposal_ingress(&self) -> Option<LinearProposalIngressConfig> {
        self.linear_proposal_ingress.clone()
    }
}

impl fmt::Debug for ModelServingHooks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelServingHooks")
            .field(
                "generation_receipt",
                &self.generation_receipt.as_ref().map(|_| "configured"),
            )
            .field(
                "linear_proposal_ingress",
                &self.linear_proposal_ingress.as_ref().map(|_| "configured"),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use anyhow::Result;

    use crate::{
        GenerationAbort, GenerationCommit, GenerationReceipt, GenerationReceiptSink,
        GenerationStart, LinearProposal, LinearProposalIngress, LinearProposalQuery,
        LinearProposalReceipt,
    };

    use super::*;

    struct ReceiptSink;

    impl GenerationReceiptSink for ReceiptSink {
        fn begin(&self, _start: &GenerationStart) -> Result<()> {
            Ok(())
        }
        fn committed(&self, _commit: &GenerationCommit) -> Result<()> {
            Ok(())
        }
        fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
            Ok(())
        }
        fn record(&self, _receipt: &GenerationReceipt) -> Result<()> {
            Ok(())
        }
    }

    struct ProposalIngress;

    impl LinearProposalIngress for ProposalIngress {
        fn propose(&self, _query: LinearProposalQuery) -> Result<Option<LinearProposal>> {
            Ok(None)
        }

        fn report(&self, _receipt: &LinearProposalReceipt) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn neutral_hooks_are_absent_by_default_and_preserved_when_injected() {
        let empty = ModelServingHooks::default();
        assert!(empty.generation_receipt().is_none());
        assert!(empty.linear_proposal_ingress().is_none());

        let configured = ModelServingHooks::new(
            GenerationReceiptConfig::new(Arc::new(ReceiptSink)),
            LinearProposalIngressConfig::new(
                Arc::new(ProposalIngress),
                Duration::from_millis(4),
                32,
            )
            .unwrap(),
        );
        assert!(configured.generation_receipt().is_some());
        assert!(configured.linear_proposal_ingress().is_some());
        assert!(configured.clone().generation_receipt().is_some());
    }

    #[test]
    fn hooks_can_be_configured_independently() {
        let receipt = ModelServingHooks::default()
            .with_generation_receipt(GenerationReceiptConfig::new(Arc::new(ReceiptSink)));
        assert!(receipt.generation_receipt().is_some());
        assert!(receipt.linear_proposal_ingress().is_none());

        let proposal = ModelServingHooks::default().with_linear_proposal_ingress(
            LinearProposalIngressConfig::new(
                Arc::new(ProposalIngress),
                Duration::from_millis(4),
                32,
            )
            .unwrap(),
        );
        assert!(proposal.generation_receipt().is_none());
        assert!(proposal.linear_proposal_ingress().is_some());
    }
}
