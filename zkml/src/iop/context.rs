use crate::{
    activation::{Activation, ActivationCtx, Relu},
    iop::precommit::{self, PolyID},
    lookup,
    model::{Layer, Model}, quantization::QuantInfo, BIT_LEN,
};
use anyhow::Context as CC;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::virtual_poly::VPAuxInfo;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StepInfo<E> {
    Dense(DenseInfo<E>),
    Activation(ActivationInfo),
}

/// Holds the poly info for the polynomials representing each matrix in the dense layers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseInfo<E> {
    pub poly_id: PolyID,
    pub poly_aux: VPAuxInfo<E>,
    pub quant_info: QuantInfo<BIT_LEN>,
}
/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationInfo {
    pub poly_id: PolyID,
    pub num_vars: usize,
    pub multiplicity_poly_id: PolyID,
    pub multiplicity_num_vars: usize,
}

impl<E> StepInfo<E> {
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::Activation(_) => "Activation".to_string(),
        }
    }
}

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: Vec<StepInfo<E>>,
    /// Context related to the commitment and accumulation of claims related to the weights of model.
    /// This part contains the commitment of the weights.
    pub weights: precommit::Context<E>,

    /// Context holding the lookup tables for activation, e.g. the MLEs of the input and output columns for
    /// RELU for example
    pub activation: ActivationCtx<E>,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Generates a context to give to the verifier that contains informations about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    pub fn generate(model: &Model) -> anyhow::Result<Self> {
        let mut last_output_size = model.first_output_shape()[0];
        let mut current_multiplicity_poly_id = model.layer_count();
        let auxs = model
            .layers()
            .map(|(id, layer)| {
                match layer {
                    Layer::Dense(matrix) => {
                        // construct dimension of the polynomial given to the sumcheck
                        let ncols = matrix.ncols();
                        last_output_size = matrix.nrows();
                        // each poly is only two polynomial right now: matrix and vector
                        // for matrix, each time we fix the variables related to rows so we are only left
                        // with the variables related to columns
                        let matrix_num_vars = ncols.ilog2() as usize;
                        let vector_num_vars = matrix_num_vars;
                        // there is only one product (i.e. quadratic sumcheck)
                        StepInfo::Dense(DenseInfo {
                            poly_id: id,
                            poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                                matrix_num_vars,
                                vector_num_vars,
                            ]]),
                            quant_info: QuantInfo::<BIT_LEN>::default(),
                        })
                    }
                    Layer::Activation(Activation::Relu(_)) => {
                        let multiplicity_poly_id = current_multiplicity_poly_id;
                        current_multiplicity_poly_id += 1;
                        StepInfo::Activation(ActivationInfo {
                            poly_id: id,
                            num_vars: last_output_size.ilog2() as usize,
                            multiplicity_poly_id,
                            multiplicity_num_vars: Relu::num_vars(),
                        })
                    }
                }
            })
            .collect_vec();
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        let activation = ActivationCtx::new();
        Ok(Self {
            steps_info: auxs.into_iter().rev().collect_vec(),
            weights: commit_ctx,
            activation,
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for steps in self.steps_info.iter() {
            match steps {
                StepInfo::Dense(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    info.poly_aux.write_to_transcript(t);
                }
                StepInfo::Activation(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                    t.append_field_element(&E::BaseField::from(info.multiplicity_poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.multiplicity_num_vars as u64));
                }
            }
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}
