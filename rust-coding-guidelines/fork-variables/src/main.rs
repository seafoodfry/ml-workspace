use nix::{
    sys::wait::waitpid,
    unistd::{fork, ForkResult},
};

fn main() {
    match unsafe { fork() } {
        Ok(ForkResult::Parent { child, .. }) => {
            println!("[parent] child has pid {child}");
            waitpid(child, None).expect("error waiting for child");
        }
        Ok(ForkResult::Child) => {
            println!("[child] im the child");
        }
        Err(error) => {
            eprintln!("error in fork(): {error}");
        }
    }
}
