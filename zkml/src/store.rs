//! PPs and scaled models KV storage.

use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher, PolynomialCommitmentScheme};
#[doc(inline)]
pub use object_store::{ClientOptions, aws::AmazonS3Builder};

use crate::{
    Element,
    model::Model,
    quantization::{ModelMetadata, ScalingStrategyKind},
};
use anyhow::{Context, bail};
use object_store::{ObjectStore, PutPayload, aws::AmazonS3, path::Path};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, future::Future};

#[derive(Debug)]
pub struct ParamsIndex<'a> {
    model_file_hash: &'a str,
}

#[derive(Debug)]
pub struct ModelIndex<'a> {
    model_file_hash: &'a str,
    scaling_strategy: ScalingStrategyKind,
    scaling_input_hash: Option<&'a str>,
}

type F = GoldilocksExt2;
type Pcs = Basefold<F, BasefoldRSParams<Hasher>>;

#[derive(Clone, Serialize, Deserialize)]
pub struct Params {
    pub prover: <Pcs as PolynomialCommitmentScheme<F>>::ProverParam,
    pub verifier: <Pcs as PolynomialCommitmentScheme<F>>::VerifierParam,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ScaledModel {
    model: Model<Element>,
    model_metadata: ModelMetadata,
}

pub trait Store {
    /// Try to get the params from store. If not present, initialize the value with the given function, store it and return.
    fn get_or_init_params_with<F>(
        &mut self,
        index: ParamsIndex<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<Params>> + Send
    where
        F: Fn() -> Params + Send + Sync;

    /// Try to get the model from store. If not present, initialize the value with the given function, store it and return.
    fn get_or_init_model_with<F>(
        &mut self,
        index: ModelIndex<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: Fn() -> ScaledModel + Send + Sync;
}

/// AWS S3 store for prod.
#[derive(Clone, derive_more::From)]
pub struct S3Store(#[from] AmazonS3);

impl Store for S3Store {
    fn get_or_init_params_with<F>(
        &mut self,
        index: ParamsIndex<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<Params>> + Send
    where
        F: Fn() -> Params + Send + Sync,
    {
        async move {
            let key = params_key(index);
            let S3Store(store) = self;
            let location = Path::parse(&key)?;
            match store.get(&location).await {
                Ok(result) => {
                    let bytes = result.bytes().await?;
                    let value = serde_json::from_slice::<Params>(&bytes)
                        .context("Decoding params value from S3")?;
                    return Ok(value);
                }
                Err(object_store::Error::NotFound { .. }) => {
                    let value = init();
                    let value_bytes: Vec<u8> = serde_json::to_vec(&value)
                        .expect("Must be able to serialize params to store");
                    store.put(&location, PutPayload::from(value_bytes)).await?;
                    Ok(value)
                }
                Err(e) => {
                    bail!(e);
                }
            }
        }
    }

    fn get_or_init_model_with<F>(
        &mut self,
        index: ModelIndex<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: Fn() -> ScaledModel + Send + Sync,
    {
        async move {
            let key = model_key(index);
            let S3Store(store) = self;
            let location = Path::parse(&key)?;
            match store.get(&location).await {
                Ok(result) => {
                    let bytes = result.bytes().await?;
                    let value = serde_json::from_slice::<ScaledModel>(&bytes)
                        .context("Decoding scaled model value from S3")?;
                    return Ok(value);
                }
                Err(object_store::Error::NotFound { .. }) => {
                    let value = init();
                    let value_bytes: Vec<u8> = serde_json::to_vec(&value)
                        .expect("Must be able to serialize scaled model to store");
                    store.put(&location, PutPayload::from(value_bytes)).await?;
                    Ok(value)
                }
                Err(e) => {
                    bail!(e);
                }
            }
        }
    }
}

/// In-memory store for testing.
// TODO consider using object_store::memory::InMemory
pub struct MemStore {
    pps: HashMap<Key, Params>,
    models: HashMap<Key, ScaledModel>,
}

impl Store for MemStore {
    fn get_or_init_params_with<F>(
        &mut self,
        index: ParamsIndex<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<Params>> + Send
    where
        F: Fn() -> Params + Send + Sync,
    {
        async move {
            let key = params_key(index);
            let value = match self.pps.get(&key) {
                Some(value) => value.clone(),
                None => {
                    let value = init();
                    self.pps.insert(key, value.clone());
                    value
                }
            };
            Ok(value)
        }
    }

    fn get_or_init_model_with<F>(
        &mut self,
        index: ModelIndex<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: Fn() -> ScaledModel + Send + Sync,
    {
        async move {
            let key = model_key(index);
            let value = match self.models.get(&key) {
                Some(value) => value.clone(),
                None => {
                    let value = init();
                    self.models.insert(key, value.clone());
                    value
                }
            };
            Ok(value)
        }
    }
}

type Key = Path;

#[derive(derive_more::Display)]
enum KeyKind {
    /// Proving parameters
    Params,
    /// Scaled model
    Model,
}

/// A store key for parameters
fn params_key(ParamsIndex { model_file_hash }: ParamsIndex<'_>) -> Key {
    let prefix = KeyKind::Params.to_string();
    let prefix = prefix.as_str();
    Path::from_iter([prefix, env!("CARGO_PKG_VERSION"), model_file_hash])
}

/// A store key for a scaled model
fn model_key(
    ModelIndex {
        model_file_hash,
        scaling_strategy,
        scaling_input_hash,
    }: ModelIndex<'_>,
) -> Key {
    let prefix = KeyKind::Model.to_string();
    let prefix = prefix.as_str();
    let scaling_strategy = scaling_strategy.to_string();
    let scaling_strategy = scaling_strategy.as_str();
    match scaling_input_hash {
        Some(scaling_input_hash) => Path::from_iter([
            prefix,
            model_file_hash,
            scaling_strategy,
            scaling_input_hash,
        ]),
        None => Path::from_iter([prefix, model_file_hash, scaling_strategy]),
    }
}
