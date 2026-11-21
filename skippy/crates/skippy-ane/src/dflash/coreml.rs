//! Rust bindings for the stateful Core ML draft model.

use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use block2::StackBlock;
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLState,
};
use objc2_foundation::{
    NSArray, NSDictionary, NSNumber, NSOperatingSystemVersion, NSProcessInfo, NSString, NSURL,
};

use super::super::profile::DFlashProfile;

/// The owning DFlash mutex serializes calls, including calls sharing an MLState.
pub(super) struct CoreMlDraft {
    model: Retained<MLModel>,
    states: Vec<Retained<MLState>>,
    context: usize,
    rows: usize,
    feature_width: usize,
    anchor_width: usize,
    rotary_width: usize,
}

impl CoreMlDraft {
    pub(super) fn open(
        path: &Path,
        lanes: usize,
        context: usize,
        rows: usize,
        profile: &DFlashProfile,
    ) -> Result<Self> {
        if !NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
            majorVersion: 15,
            minorVersion: 0,
            patchVersion: 0,
        }) {
            bail!("ANE DFlash requires macOS 15 or newer");
        }
        if lanes == 0 || context < rows || rows < 2 {
            bail!("invalid ANE draft model dimensions");
        }
        let path = path.to_str().context("ANE draft model path is not UTF-8")?;
        autoreleasepool(|_| {
            let url = NSURL::fileURLWithPath(&NSString::from_str(path));
            let configuration = unsafe { MLModelConfiguration::new() };
            unsafe { configuration.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
            let model = unsafe {
                MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration)
            }
            .map_err(|error| anyhow!("load Core ML draft: {error:?}"))?;
            validate_shape(&model, context, rows, profile)?;
            let states = (0..lanes).map(|_| unsafe { model.newState() }).collect();
            Ok(Self {
                model,
                states,
                context,
                rows,
                feature_width: profile
                    .feature_width()
                    .context("invalid DFlash feature width")?,
                anchor_width: profile.width,
                rotary_width: profile.rotary_width,
            })
        })
    }

    pub(super) fn reset(&mut self, lane: usize) -> Result<()> {
        let state = self
            .states
            .get_mut(lane)
            .context("ANE draft lane is invalid")?;
        *state = unsafe { self.model.newState() };
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Core ML's fixed model inputs form the ABI"
    )]
    pub(super) fn predict(
        &mut self,
        lane: usize,
        features: &[u16],
        write: &[u16],
        ctx_cos: &[u16],
        ctx_sin: &[u16],
        anchor: &[u16],
        q_cos: &[u16],
        q_sin: &[u16],
        mask: &[u16],
    ) -> Result<Vec<i32>> {
        let state = self.states.get(lane).context("ANE draft lane is invalid")?;
        let rows = self.rows;
        let context = self.context;
        let feature_width = self.feature_width;
        let anchor_width = self.anchor_width;
        let rotary_width = self.rotary_width;
        autoreleasepool(|_| {
            let inputs = [
                ("features", &[1, feature_width, 1, rows][..], features),
                ("write", &[rows, context], write),
                ("ctx_cos", &[rows, rotary_width], ctx_cos),
                ("ctx_sin", &[rows, rotary_width], ctx_sin),
                ("anchor", &[1, anchor_width, 1, 1], anchor),
                ("q_cos", &[rows, rotary_width], q_cos),
                ("q_sin", &[rows, rotary_width], q_sin),
                ("mask", &[rows, context + rows], mask),
            ];
            let names = inputs
                .iter()
                .map(|(name, _, _)| NSString::from_str(name))
                .collect::<Vec<_>>();
            let values = inputs
                .iter()
                .map(|(_, shape, data)| array_value(shape, data))
                .collect::<Result<Vec<_>>>()?;
            let name_refs = names.iter().map(|name| &**name).collect::<Vec<_>>();
            let value_refs = values
                .iter()
                .map(|value| value.as_ref())
                .collect::<Vec<&AnyObject>>();
            let dictionary = NSDictionary::from_slices(&name_refs, &value_refs);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dictionary,
                )
            }
            .map_err(|error| anyhow!("create ANE draft input: {error:?}"))?;
            let output = unsafe {
                self.model.predictionFromFeatures_usingState_error(
                    ProtocolObject::<dyn MLFeatureProvider>::from_ref(&*provider),
                    state,
                )
            }
            .map_err(|error| anyhow!("Core ML DFlash prediction failed: {error:?}"))?;
            let tokens = unsafe { output.featureValueForName(&NSString::from_str("tokens")) }
                .context("ANE draft omitted tokens output")?;
            let tokens = unsafe { tokens.multiArrayValue() }
                .context("ANE draft tokens output is not an array")?;
            if unsafe { tokens.dataType() } != MLMultiArrayDataType::Int32
                || usize::try_from(unsafe { tokens.count() })? != rows - 1
            {
                bail!("ANE draft returned an unexpected token tensor");
            }
            let shape = unsafe { tokens.shape() };
            if shape.len() != 3
                || shape.objectAtIndex(0).unsignedIntegerValue() != 1
                || shape.objectAtIndex(1).unsignedIntegerValue() != 1
                || shape.objectAtIndex(2).unsignedIntegerValue() != rows - 1
            {
                bail!("ANE draft returned an unexpected token shape");
            }
            let strides = unsafe { tokens.strides() };
            if strides.len() != 3 {
                bail!("ANE draft returned an unexpected token stride layout");
            }
            let stride = usize::try_from(strides.objectAtIndex(2).integerValue())?;
            if stride == 0 {
                bail!("ANE draft returned a zero token stride");
            }
            let copied = RefCell::new(None);
            let copy_destination = &copied;
            let copy = StackBlock::new(move |source: std::ptr::NonNull<std::ffi::c_void>, size| {
                let Ok(size) = usize::try_from(size) else {
                    return;
                };
                if size % std::mem::size_of::<i32>() != 0
                    || size > isize::MAX as usize
                    || !source.as_ptr().cast::<i32>().is_aligned()
                {
                    return;
                }
                // SAFETY: Core ML supplies an aligned Int32 buffer that remains valid
                // for the synchronous block call; `size` is its byte length.
                let storage = unsafe {
                    std::slice::from_raw_parts(
                        source.as_ptr().cast::<i32>(),
                        size / std::mem::size_of::<i32>(),
                    )
                };
                copy_destination.replace(read_strided_tokens(storage, rows - 1, stride));
            });
            unsafe { tokens.getBytesWithHandler(&copy) };
            copied
                .into_inner()
                .context("ANE draft token buffer is shorter than its strides require")
        })
    }
}

fn read_strided_tokens(storage: &[i32], count: usize, stride: usize) -> Option<Vec<i32>> {
    if stride == 0 {
        return None;
    }
    if count == 0 {
        return Some(Vec::new());
    }
    let last = (count - 1).checked_mul(stride)?;
    if last >= storage.len() {
        return None;
    }
    if stride == 1 {
        Some(storage[..count].to_vec())
    } else {
        Some((0..count).map(|index| storage[index * stride]).collect())
    }
}

fn array_value(shape: &[usize], data: &[u16]) -> Result<Retained<MLFeatureValue>> {
    let count = shape.iter().product::<usize>();
    if count != data.len() {
        bail!("ANE draft input has the wrong number of elements");
    }
    let dimensions = shape
        .iter()
        .map(|&dim| NSNumber::new_usize(dim))
        .collect::<Vec<_>>();
    let dimensions = NSArray::from_retained_slice(&dimensions);
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &dimensions,
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(|error| anyhow!("allocate ANE draft input: {error:?}"))?;
    let copy = StackBlock::new(|destination: std::ptr::NonNull<std::ffi::c_void>, _, _| {
        // SAFETY: Core ML invokes this block synchronously with a destination of
        // exactly `count` Float16 values; the source slice remains live.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), destination.as_ptr().cast(), count);
        }
    });
    unsafe { array.getMutableBytesWithHandler(&copy) };
    Ok(unsafe { MLFeatureValue::featureValueWithMultiArray(&array) })
}

fn validate_shape(
    model: &MLModel,
    context: usize,
    rows: usize,
    profile: &DFlashProfile,
) -> Result<()> {
    let description = unsafe { model.modelDescription() };
    let inputs = unsafe { description.inputDescriptionsByName() };
    let feature_width = profile
        .feature_width()
        .context("invalid DFlash feature width")?;
    for (name, expected) in [
        ("features", &[1, feature_width, 1, rows][..]),
        ("write", &[rows, context]),
        ("ctx_cos", &[rows, profile.rotary_width]),
        ("ctx_sin", &[rows, profile.rotary_width]),
        ("anchor", &[1, profile.width, 1, 1]),
        ("q_cos", &[rows, profile.rotary_width]),
        ("q_sin", &[rows, profile.rotary_width]),
        ("mask", &[rows, context + rows][..]),
    ] {
        let feature = inputs
            .objectForKey(&NSString::from_str(name))
            .with_context(|| format!("ANE draft lacks {name} input"))?;
        let constraint = unsafe { feature.multiArrayConstraint() }
            .with_context(|| format!("ANE draft {name} is not an array"))?;
        let shape = unsafe { constraint.shape() };
        if shape.len() != expected.len()
            || expected
                .iter()
                .enumerate()
                .any(|(index, &size)| shape.objectAtIndex(index).unsignedIntegerValue() != size)
        {
            bail!("ANE draft {name} input shape does not match the session");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_strided_tokens;

    #[test]
    fn token_reads_follow_last_dimension_stride() {
        assert_eq!(
            read_strided_tokens(&[11, 22, 33], 3, 1),
            Some(vec![11, 22, 33])
        );
        assert_eq!(
            read_strided_tokens(&[11, -1, 22, -1, 33], 3, 2),
            Some(vec![11, 22, 33])
        );
        assert_eq!(read_strided_tokens(&[11, -1, 22], 3, 2), None);
    }
}
