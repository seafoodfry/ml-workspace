use nix::unistd::{fork, ForkResult};
use rand::{thread_rng, Rng};

mod child;
mod parent;

fn main() {
    let random_sleep_time: u32 = thread_rng().gen_range(0..=10);
    println!(
        "[main] child thread will sleep for {:.3}",
        random_sleep_time
    );

    match unsafe { fork() } {
        Ok(ForkResult::Parent { child, .. }) => parent::parent_process(child, random_sleep_time),
        Ok(ForkResult::Child) => child::child_process(random_sleep_time),
        Err(error) => eprintln!("error in fork(): {error}"),
    }
}
