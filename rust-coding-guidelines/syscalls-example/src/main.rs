use std::os::raw::{c_char, c_long};

const STDOUT_FILENO: i32 = 1;

extern "C" {
    // See https://docs.rs/libc/latest/libc/fn.syscall.html , this is pretty much the same thing
    // libc does.
    fn syscall(number: c_long, ...) -> c_long;
}

fn main() {
    #[cfg(target_os = "linux")]
    {
        let msg = "Hello syscalls, this is Rust!\n";
        let result = unsafe {
            syscall(
                libc::SYS_write,
                STDOUT_FILENO,
                msg.as_ptr() as *const c_char,
                msg.len() as c_long,
            )
        };
        println!("length of Rust message: {}", msg.len());
        println!("Syscall result: {result}")
    }

    #[cfg(target_os = "macos")]
    {
        println!("This example was not meant to run on a non-linux os. Do make docker-run")
    }
}
