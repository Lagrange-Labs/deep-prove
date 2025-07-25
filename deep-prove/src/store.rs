//! PPs and scaled models KV storage.
#![allow(clippy::manual_async_fn)]

use anyhow::{Context, bail};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher, PolynomialCommitmentScheme};
#[doc(inline)]
pub use object_store::{
    ClientOptions,
    aws::{AmazonS3, AmazonS3Builder},
};
use object_store::{GetOptions, ObjectStore, PutPayload, path::Path};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::HashMap,
    env,
    future::Future,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tempfile::TempDir;
use tokio::fs;
use zkml::{
    Element,
    model::Model,
    quantization::{ModelMetadata, ScalingStrategyKind},
};

#[derive(Debug, Clone, Copy)]
pub struct ParamsKey<'a> {
    pub model_file_hash: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelKey<'a> {
    pub model_file_hash: &'a str,
    pub scaling_strategy: ScalingStrategyKind,
    pub scaling_input_hash: Option<&'a str>,
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
    pub model: Model<Element>,
    pub model_metadata: ModelMetadata,
}

pub trait Store {
    /// Try to get the params from store.
    fn get_params(
        &mut self,
        key: ParamsKey<'_>,
    ) -> impl Future<Output = anyhow::Result<Option<Params>>> + Send;

    /// Store the params.
    fn insert_params(
        &mut self,
        key: ParamsKey<'_>,
        params: Params,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    /// Try to get the model from store. If not present, initialize the value with the given function, store it and return.
    fn get_or_init_model_with<F, FR>(
        &mut self,
        key: ModelKey<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: FnOnce() -> FR + Send,
        FR: Future<Output = anyhow::Result<ScaledModel>> + Send;
}

/// AWS S3 store for prod.
#[derive(Clone, derive_more::From)]
pub struct S3Store {
    store: AmazonS3,
    fs_cache: Option<Arc<TempDir>>,
}

impl From<AmazonS3> for S3Store {
    fn from(store: AmazonS3) -> Self {
        S3Store {
            store,
            fs_cache: None,
        }
    }
}

impl S3Store {
    pub fn with_fs_cache(mut self, fs_cache_dir: PathBuf) -> Self {
        self.fs_cache = Some(Arc::new(
            TempDir::new_in(fs_cache_dir).expect("able to setup an S3 store cache in a temp dir"),
        ));
        self
    }
}

impl Store for S3Store {
    fn get_params(
        &mut self,
        key: ParamsKey<'_>,
    ) -> impl Future<Output = anyhow::Result<Option<Params>>> + Send {
        async move {
            let key = params_key(key);
            let S3Store { store, fs_cache } = self;

            // Try read from FS cache first
            let cache_path = fs_cache
                .as_ref()
                .map(|cache| cache.path().join(key.to_string()));
            if let Some(path) = &cache_path {
                if fs::try_exists(&path).await.context("access FS cache")? {
                    let bytes = fs::read(path).await?;
                    let value = serde_json::from_slice::<Params>(&bytes)
                        .context("decoding params value from FS cache")?;
                    return Ok(Some(value));
                }
            }
            match store.get(&key).await {
                Ok(result) => {
                    let bytes = result.bytes().await?;
                    let value = serde_json::from_slice::<Params>(&bytes)
                        .context("decoding params value from S3")?;

                    // Cache to FS
                    if let Some(path) = cache_path {
                        fs::create_dir_all(&path)
                            .await
                            .context("create FS cache dirs")?;
                        fs::write(&path, &bytes)
                            .await
                            .context("write params to FS cache")?;
                    }

                    Ok(Some(value))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => {
                    bail!(e);
                }
            }
        }
    }

    fn insert_params(
        &mut self,
        key: ParamsKey<'_>,
        params: Params,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        async move {
            let value_bytes: Vec<u8> =
                serde_json::to_vec(&params).context("serializing params to store")?;
            let key = params_key(key);
            let S3Store { store, fs_cache } = self;

            // Write to FS cache first
            let cache_path = fs_cache
                .as_ref()
                .map(|cache| cache.path().join(key.to_string()));
            if let Some(path) = cache_path {
                if !fs::try_exists(&path).await.context("access FS cache")? {
                    fs::create_dir_all(&path)
                        .await
                        .context("create FS cache dirs")?;
                    fs::write(&path, &value_bytes)
                        .await
                        .context("write params to FS cache")?;
                }
            }

            if store
                .get_opts(
                    &key,
                    GetOptions {
                        head: true,
                        ..Default::default()
                    },
                )
                .await
                .is_ok()
            {
                bail!("trying to insert params with {key} that is already present")
            }
            store
                .put(&key, PutPayload::from(value_bytes))
                .await
                .context("storing generated params in S3 store")?;
            Ok(())
        }
    }

    fn get_or_init_model_with<F, FR>(
        &mut self,
        key: ModelKey<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: FnOnce() -> FR + Send,
        FR: Future<Output = anyhow::Result<ScaledModel>> + Send,
    {
        async move {
            let key = model_key(key);
            let S3Store { store, fs_cache } = self;

            // Try read from FS cache first
            let cache_path = fs_cache
                .as_ref()
                .map(|cache| cache.path().join(key.to_string()));
            if let Some(path) = &cache_path {
                if fs::try_exists(&path).await.context("access FS cache")? {
                    let bytes = fs::read(path).await?;
                    let value = serde_json::from_slice::<ScaledModel>(&bytes)
                        .context("decoding scaled model value from FS cache")?;
                    return Ok(value);
                }
            }

            match store.get(&key).await {
                Ok(result) => {
                    let bytes = result.bytes().await?;
                    let value = serde_json::from_slice::<ScaledModel>(&bytes)
                        .context("decoding scaled model value from S3")?;

                    // Cache to FS
                    if let Some(path) = cache_path {
                        fs::create_dir_all(&path)
                            .await
                            .context("create FS cache dirs")?;
                        fs::write(&path, &bytes)
                            .await
                            .context("write params to FS cache")?;
                    }

                    Ok(value)
                }
                Err(object_store::Error::NotFound { .. }) => {
                    let value = init().await?;
                    let value_bytes: Vec<u8> =
                        serde_json::to_vec(&value).context("serializing scaled model to store")?;

                    // Write to FS cache first
                    if let Some(path) = cache_path {
                        fs::create_dir_all(&path)
                            .await
                            .context("create FS cache dirs")?;
                        fs::write(&path, &value_bytes)
                            .await
                            .context("write params to FS cache")?;
                    }

                    store
                        .put(&key, PutPayload::from(value_bytes))
                        .await
                        .context("storing generated params in S3 store")?;
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
#[derive(Clone, Default)]
pub struct MemStore {
    pps: Arc<Mutex<HashMap<Key, Params>>>,
    models: Arc<Mutex<HashMap<Key, ScaledModel>>>,
}

#[derive(Clone, Default)]
pub struct MemStoreInner {}

impl Store for MemStore {
    fn get_params(
        &mut self,
        key: ParamsKey<'_>,
    ) -> impl Future<Output = anyhow::Result<Option<Params>>> + Send {
        async move {
            let key = params_key(key);
            let guard = self.pps.lock().unwrap();
            Ok(guard.get(&key).cloned())
        }
    }

    fn insert_params(
        &mut self,
        key: ParamsKey<'_>,
        params: Params,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        async move {
            let key = params_key(key);
            let mut guard = self.pps.lock().unwrap();
            guard.insert(key, params);
            Ok(())
        }
    }

    fn get_or_init_model_with<F, FR>(
        &mut self,
        key: ModelKey<'_>,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: FnOnce() -> FR + Send,
        FR: Future<Output = anyhow::Result<ScaledModel>> + Send,
    {
        async move {
            let key = model_key(key);
            let get_result = {
                let guard = self.models.lock().unwrap();
                guard.get(&key).cloned()
            };
            let value = match get_result {
                Some(value) => value,
                None => {
                    let value = init().await?;
                    let mut guard = self.models.lock().unwrap();
                    guard.insert(key, value.clone());
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
fn params_key(ParamsKey { model_file_hash }: ParamsKey<'_>) -> Key {
    let prefix = KeyKind::Params.to_string();
    let prefix = prefix.as_str();
    let pkg_major_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .map(|version| Cow::from(version.major.to_string()))
        .unwrap_or_else(|_| Cow::from("version-unknown"));
    Path::from_iter([prefix, &pkg_major_version, model_file_hash])
}

/// A store key for a scaled model
fn model_key(
    ModelKey {
        model_file_hash,
        scaling_strategy,
        scaling_input_hash,
    }: ModelKey<'_>,
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
