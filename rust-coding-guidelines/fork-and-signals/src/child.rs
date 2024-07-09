use nix::unistd::{getpid, getppid};
use std::thread;
use std::time::Duration;

pub fn child_process(random_sleep_time: u32) {
    let parent_pid = getppid();
    println!("[child] parent pid: {parent_pid}");
    let pid = getpid();
    println!("[child] current pid: {pid}");

    println!("[child] going to sleep for {random_sleep_time} seconds");
    thread::sleep(Duration::from_secs(random_sleep_time.into()));
    println!("[child] waking up now");
}
