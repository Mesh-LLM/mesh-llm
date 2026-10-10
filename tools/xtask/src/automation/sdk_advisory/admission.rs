use super::error::{Checked, Rejected};
use super::identity::RunId;
use super::input::{ManualEvent, ProducerRun, WorkflowEvent, parse};
use super::rows::{ProducerWorkflow, ProductRow};

pub(super) const REPOSITORY: &str = "Mesh-LLM/mesh-llm";

pub(super) struct Controller<'a> {
    pub(super) repository: &'a str,
    pub(super) reference: &'a str,
}

pub(super) enum Trigger<'a> {
    WorkflowRun(&'a [u8]),
    Manual(&'a [u8]),
}

#[derive(Debug)]
pub(super) struct AdmittedProducer {
    run: ProducerRun,
    rows: Vec<ProductRow>,
    workflow: ProducerWorkflow,
}

impl AdmittedProducer {
    pub(super) const fn run(&self) -> &ProducerRun {
        &self.run
    }

    pub(super) fn rows(&self) -> &[ProductRow] {
        &self.rows
    }

    pub(super) const fn workflow(&self) -> ProducerWorkflow {
        self.workflow
    }
}

pub(super) fn admit(
    controller: &Controller<'_>,
    trigger: Trigger<'_>,
    run_bytes: &[u8],
) -> Checked<AdmittedProducer> {
    if controller.repository != REPOSITORY || controller.reference != "refs/heads/main" {
        return Err(Rejected::Controller);
    }
    let run: ProducerRun = parse(run_bytes)?;
    let workflow = authorize_run(&run)?;
    let rows = match trigger {
        Trigger::WorkflowRun(bytes) => {
            let event: WorkflowEvent = parse(bytes)?;
            if event.action != "completed" {
                return Err(Rejected::Event);
            }
            if event.repository != run.repository {
                return Err(Rejected::Repository);
            }
            if event.workflow_run != run {
                return Err(Rejected::RunIdentity);
            }
            workflow.rows().to_vec()
        }
        Trigger::Manual(bytes) => {
            let event: ManualEvent = parse(bytes)?;
            if event.repository != run.repository {
                return Err(Rejected::Repository);
            }
            let requested_id =
                RunId::parse(&event.inputs.producer_run_id).map_err(|_| Rejected::RunIdentity)?;
            if requested_id != run.id {
                return Err(Rejected::RunIdentity);
            }
            if !workflow.rows().contains(&event.inputs.product_row) {
                return Err(Rejected::RowProducer);
            }
            vec![event.inputs.product_row]
        }
    };
    Ok(AdmittedProducer {
        run,
        rows,
        workflow,
    })
}

fn authorize_run(run: &ProducerRun) -> Checked<ProducerWorkflow> {
    if run.repository.full_name != REPOSITORY || run.head_repository != run.repository {
        return Err(Rejected::Repository);
    }
    if run.event != "push" {
        return Err(Rejected::ProducerEvent);
    }
    if run.head_branch != "main" {
        return Err(Rejected::Branch);
    }
    if run.status != "completed" || run.conclusion.as_deref() != Some("success") {
        return Err(Rejected::Conclusion);
    }
    ProducerWorkflow::parse(&run.name, &run.path)
}
