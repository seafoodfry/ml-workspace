# Benchmarking Intro

A quick idea that we came up with to learn how to benchmark Rust code was in comparing these two version of the
following code:

```rust
use std::env as p_env;

fn clean_env_vars() {
    for (key, _) in p_env::vars() {
        p_env::remove_var(key);
    }
}
```

```rust
use std::env as p_env;

fn clean_env_vars() {
    p_env::vars().for_each(|(key, _)| p_env::remove_var(key));
}
```

For this task we will use
[github.com/bheisler/criterion.rs](https://github.com/bheisler/criterion.rs)

All we need to do to run the code is the following command:
```
cargo bench
```

This will generate a very informative and beautiful report that you can see via:
```
open ../../target/criterion/report/index.html
```