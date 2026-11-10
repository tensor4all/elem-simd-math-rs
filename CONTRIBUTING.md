<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to elem-simd-math-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `elem-simd-math-rs`:

## Local facts

- **MSRV**: `rust-version = "1.89"` in `Cargo.toml`. CI pins `stable` and has no `msrv` job, so the claim is not
  checked by CI.
- **Feature combinations CI builds**: `default` and `--no-default-features`, each built and tested — `cargo clippy
  -j 16 --all-targets -- -D warnings` plus `cargo test -j 16 --release` for the default set, and the same with
  `--no-default-features` for the other.
- **Platforms claimed**: `ubuntu-latest` for the single `check` job; no macOS or Windows job exists.
- **Local gate**: `cargo test --release`.

  The ULP suites measure against MPFR through `rug`, whose `gmp-mpfr-sys` builds GMP and MPFR itself and therefore
  needs `m4` on the machine. CI installs the toolchain first:

  ```sh
  sudo apt-get update && sudo apt-get install -y libgmp-dev libmpfr-dev m4
  cargo test --release
  ```

  `m4` is the step people forget; without it the oracle does not build and the ULP suites cannot run.

## Anything here that departs from the shared rules

One does: the `msrv` gate is not run — `rust-version = "1.89"` is declared but nothing builds on 1.89, so the claim
can drift with a dependency bump. That is a gap to close, not an exemption from the other gates.
