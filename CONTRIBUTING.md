# Contributing to gadget-ng

## Getting Started

1. Install Rust 1.85+ via [rustup](https://rustup.rs)
2. Clone: `git clone https://github.com/cavazquez/gadget-ng.git`
3. Build: `cargo build --release -p gadget-ng-cli`
4. Test: `cargo test --workspace`

## Development Flow

1. Fork and create a feature branch
2. Make changes following [Conventional Commits](https://www.conventionalcommits.org/)
3. Run `bash scripts/check.sh` before pushing
4. If touching physics: `bash scripts/check-physics.sh`
5. Open a PR against `main`

## Code Style

- `cargo fmt --all` enforced in CI
- `cargo clippy --workspace -- -D warnings` enforced in CI
- Unsafe code requires `// SAFETY:` comment
- Use `#[expect(clippy::lint)]` over `#[allow(...)]`

## Testing

- Quick check: `cargo test -p gadget-ng-core`
- Full check: `cargo test --workspace`
- Physics: `bash scripts/check-physics.sh`
- Deep: `cargo test -p gadget-ng-physics -- --include-ignored`

## See Also

- [`AGENTS.md`](AGENTS.md) — AI assistant guidelines
- [`docs/reports/`](docs/reports/) — Technical reports per phase
