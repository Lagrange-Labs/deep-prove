FROM rustlang/rust:nightly-slim AS builder
RUN apt-get update && apt-get install -y git protobuf-compiler libssl-dev pkg-config

WORKDIR /deep-prove-worker
COPY . .

RUN cargo install --locked --path deep-prove --bin deep-prove-worker


# Create the actual final image
FROM docker.io/library/ubuntu:22.04

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} lagrange && \
    useradd -m -u ${USER_ID} -g lagrange lagrange

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libssl3 ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cargo/bin/deep-prove-worker /usr/local/bin
EXPOSE 8080
ENV RUST_BACKTRACE=full
USER lagrange
ENTRYPOINT ["/usr/local/bin/deep-prove-worker"]
