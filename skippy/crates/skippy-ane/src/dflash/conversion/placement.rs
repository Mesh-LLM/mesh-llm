//! Check that Core ML places every costly draft operation on ANE.

use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result, anyhow, bail};
use block2::RcBlock;
use objc2::ClassType;
use objc2_core_ml::{
    MLComputePlan, MLComputeUnits, MLModelConfiguration, MLNeuralEngineComputeDevice,
};
use objc2_foundation::{NSError, NSObjectProtocol, NSString, NSURL};

#[derive(Debug)]
pub(super) struct Placement {
    pub neural_operations: usize,
    pub neural_operations_on_ane: usize,
}

pub(super) fn inspect(compiled: &Path) -> Result<Placement> {
    let path = compiled
        .to_str()
        .context("compiled model path is not UTF-8")?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(path));
    let configuration = unsafe { MLModelConfiguration::new() };
    unsafe { configuration.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
    let (tx, rx) = mpsc::channel();
    let callback = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
        let result = unsafe { plan.as_ref() }
            .ok_or_else(|| anyhow!("Core ML compute plan failed: {error:p}"))
            .and_then(count_placement);
        let _ = tx.send(result);
    });
    unsafe {
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(
            &url,
            &configuration,
            &callback,
        );
    }
    let result = rx.recv().context("Core ML compute plan did not return")??;
    if result.neural_operations == 0 || result.neural_operations != result.neural_operations_on_ane
    {
        bail!(
            "Core ML placed only {}/{} costly draft operations on ANE",
            result.neural_operations_on_ane,
            result.neural_operations
        );
    }
    Ok(result)
}

fn count_placement(plan: &MLComputePlan) -> Result<Placement> {
    let structure = unsafe { plan.modelStructure() };
    let program = unsafe { structure.program() }.context("draft is not an ML Program")?;
    let functions = unsafe { program.functions() };
    let main = functions
        .objectForKey(&NSString::from_str("main"))
        .context("draft ML Program has no main function")?;
    let block = unsafe { main.block() };
    let operations = unsafe { block.operations() };
    let mut result = Placement {
        neural_operations: 0,
        neural_operations_on_ane: 0,
    };
    for index in 0..operations.len() {
        let operation = operations.objectAtIndex(index);
        let name = unsafe { operation.operatorName() }.to_string();
        if !requires_neural_engine(&name) {
            continue;
        }
        result.neural_operations += 1;
        let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(&operation) }
            .with_context(|| format!("Core ML omitted {name} placement"))?;
        let preferred = unsafe { usage.preferredComputeDevice() };
        if preferred.isKindOfClass(MLNeuralEngineComputeDevice::class()) {
            result.neural_operations_on_ane += 1;
        }
    }
    Ok(result)
}

fn requires_neural_engine(name: &str) -> bool {
    let operator = name.rsplit('.').next().unwrap_or(name);
    ["conv", "matmul", "softmax"].contains(&operator)
}

#[cfg(test)]
mod tests {
    use super::requires_neural_engine;

    #[test]
    fn bare_and_prefixed_costly_ops_require_ane() {
        for name in ["conv", "matmul", "softmax", "ios18.conv", "ios18.softmax"] {
            assert!(requires_neural_engine(name), "{name}");
        }
        assert!(!requires_neural_engine("ios18.argmax"));
    }
}
