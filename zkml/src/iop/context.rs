use crate::{
    activation::Activation,
    iop::precommit::{self, PolyID},
    lookup::Context as LookupContext,
    model::{Layer, Model},
    pooling::{Maxpool2D, Pooling},
    quantization::Requant,
};
use anyhow::Context as CC;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::BasefoldCommitment;
use multilinear_extensions::virtual_poly::VPAuxInfo;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
/// NOTE: The context automatically appends a requant step after each dense layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum StepInfo<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    Dense(DenseInfo<E>),
    Activation(ActivationInfo),
    Requant(RequantInfo),
    Pooling(PoolingInfo), // Maxpool update
    Table(TableInfo<E>),
}

/// Holds the poly info for the polynomials representing each matrix in the dense layers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseInfo<E> {
    pub poly_id: PolyID,
    pub poly_aux: VPAuxInfo<E>,
}
/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationInfo {
    pub op: Activation,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

/// Info related to the lookup protocol necessary to requantize
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequantInfo {
    pub requant: Requant,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
// Maxpool update
pub struct PoolingInfo {
    pub poolinfo: Maxpool2D,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

/// Info related to the lookup protocol tables.
/// Here `poly_id` is the multiplicity poly for this table.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct TableInfo<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub poly_id: PolyID,
    pub num_vars: usize,
    pub table_commitment: BasefoldCommitment<E>,
}

impl<E> StepInfo<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(), // Maxpool update
            Self::Table(..) => "Table".to_string(),
        }
    }

    pub fn requires_lookup(&self) -> bool {
        match self {
            Self::Dense(..) => false,
            _ => true,
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
    /// Context holding all lookup related inforamtion
    pub lookup: LookupContext<E>,
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
        let auxs = model
            .layers()
            .map(|(id, layer)| {
                match layer {
                    Layer::Dense(matrix) => {
                        // construct dimension of the polynomial given to the sumcheck
                        let ncols = matrix.ncols_2d();
                        last_output_size = matrix.nrows_2d();
                        // each poly is only two polynomial right now: matrix and vector
                        // for matrix, each time we fix the variables related to rows so we are only left
                        // with the variables related to columns
                        let matrix_num_vars = ncols.ilog2() as usize;
                        let vector_num_vars = matrix_num_vars;
                        // there is only one product (i.e. quadratic sumcheck)
                        let dense_info = StepInfo::Dense(DenseInfo {
                            poly_id: id,
                            poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                                matrix_num_vars,
                                vector_num_vars,
                            ]]),
                        });
                        dense_info
                    }
                    Layer::Activation(Activation::Relu(relu)) => {
                        StepInfo::Activation(ActivationInfo {
                            op: Activation::Relu(*relu),
                            poly_id: id,
                            num_vars: last_output_size.ilog2() as usize,
                        })
                    }
                    Layer::Requant(info) => StepInfo::Requant(RequantInfo {
                        requant: *info,
                        poly_id: id,
                        num_vars: last_output_size.ilog2() as usize,
                    }),
                    Layer::Pooling(Pooling::Maxpool2D(info)) => StepInfo::Pooling(PoolingInfo {
                        // Maxpool update
                        poolinfo: *info,
                        poly_id: id,
                        num_vars: last_output_size.ilog2() as usize,
                    }),
                }
            })
            .collect_vec();
        println!("CTX STEP B");
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        println!("CTX STEP C");

        let lookup = LookupContext::<E>::generate(&auxs)?;

        Ok(Self {
            steps_info: auxs.into_iter().rev().collect_vec(),
            weights: commit_ctx,
            lookup,
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for steps in self.steps_info.iter() {
            match steps {
                StepInfo::Dense(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    info.poly_aux.write_to_transcript(t);
                }
                StepInfo::Requant(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                StepInfo::Activation(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                StepInfo::Pooling(info) => {
                    // Maxpool update
                    t.append_field_element(&E::BaseField::from(info.poolinfo.kernel_size as u64));
                    t.append_field_element(&E::BaseField::from(info.poolinfo.stride as u64));
                }
                StepInfo::Table(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                    t.append_field_elements(info.table_commitment.root().0.as_slice());
                }
            }
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}
