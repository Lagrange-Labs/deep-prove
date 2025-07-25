use deep_prove::store::MemStore;

use crate::{S3Args, S3Store, StoreKind, store};
use anyhow::Context;
use tracing::info;

pub mod grpc;
pub mod http;

fn instantiate_store(args: S3Args) -> anyhow::Result<StoreKind> {
    let S3Args {
        s3_region,
        s3_bucket,
        s3_endpoint,
        s3_timeout_secs,
        s3_access_key_id,
        s3_secret_access_key,
        fs_cache,
        fs_cache_dir,
    } = args;

    Ok(if s3_bucket.is_some() {
        let region = s3_region.context("gathering S3 config arguments")?;
        let timeout = std::time::Duration::from_secs(s3_timeout_secs.unwrap());
        let s3: store::AmazonS3 = store::AmazonS3Builder::new()
            .with_region(region)
            .with_bucket_name(s3_bucket.context("S3 bucket name")?)
            .with_access_key_id(s3_access_key_id.context("S3 access key ID")?)
            .with_secret_access_key(s3_secret_access_key.context("S3 secret access key")?)
            .with_endpoint(s3_endpoint.context("S3 endpoint")?)
            .with_client_options(
                store::ClientOptions::default()
                    .with_timeout(timeout)
                    .with_allow_http(true),
            )
            .build()
            .context("AWS S3 builder")?;
        let s3 = S3Store::from(s3);
        let s3 = if fs_cache {
            s3.with_fs_cache(fs_cache_dir)
        } else {
            s3
        };
        info!("using S3 store");
        StoreKind::S3(s3)
    } else {
        info!("using in-memory store");
        StoreKind::Mem(MemStore::default())
    })
}
