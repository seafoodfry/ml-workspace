# ML Workspace

### Getting Started

Install [git-secrets](https://github.com/awslabs/git-secrets) to scan for secrets.
We also make use of [github.com/rustsec/rustsec](https://github.com/rustsec/rustsec)
and [github.com/swellaby/rusty-hook](https://github.com/swellaby/rusty-hook).

```
brew install git-secrets
cargo install cargo-audit
cargo install rusty-hook
```

Then to initialize these tools on each repo we do
```
git secrets --register-aws
rusty-hook init
```
