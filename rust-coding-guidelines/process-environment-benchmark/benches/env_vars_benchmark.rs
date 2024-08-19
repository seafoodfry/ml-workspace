use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::env as p_env;
use std::time::Duration;

fn clean_env_vars_loop() {
    for (key, _) in p_env::vars() {
        p_env::remove_var(key);
    }
}

fn clean_env_vars_for_each() {
    p_env::vars().for_each(|(key, _)| p_env::remove_var(key));
}

fn setup_env_vars(num_vars: usize) {
    for i in 0..num_vars {
        p_env::set_var(format!("TEST_VAR_{}", i), "test_value");
    }
}

fn benchmark_clean_env_vars(c: &mut Criterion) {
    let num_vars = 100; // You can adjust this number

    let mut group = c.benchmark_group("clean_env_vars");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("loop", |b| {
        b.iter(|| {
            setup_env_vars(num_vars);
            black_box(clean_env_vars_loop());
        })
    });

    group.bench_function("for_each", |b| {
        b.iter(|| {
            setup_env_vars(num_vars);
            black_box(clean_env_vars_for_each());
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_clean_env_vars);
criterion_main!(benches);
