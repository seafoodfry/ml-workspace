use nix::sys::signal::{kill, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;
use std::cmp::Ordering;
use std::thread;
use std::time::Duration;

pub fn parent_process(child: Pid, random_sleep_time: u32) {
    let parent_pid = Pid::parent();
    println!("[parent] parent pid: {parent_pid}");
    let current_pid = Pid::this();
    println!("[parent] current pid: {current_pid}");
    println!("[parent] child'd pid: {child}");

    handle_child_signal(child, random_sleep_time);
    wait_for_child(child, random_sleep_time);
}

fn handle_child_signal(child: Pid, random_sleep_time: u32) {
    let base: u32 = 3;
    match random_sleep_time.cmp(&base) {
        Ordering::Greater => match kill(child, Signal::SIGSTOP) {
            Ok(_) => println!("[parent] sent stop signal to child"),
            Err(error) => eprintln!("[parent] error sending stop signal to child: {error}"),
        },
        _ => println!("[parent] child was not sent a signal"),
    }
}

fn wait_for_child(child: Pid, random_sleep_time: u32) {
    loop {
        match waitpid(
            child,
            Some(WaitPidFlag::WUNTRACED | WaitPidFlag::WCONTINUED),
        ) {
            Ok(WaitStatus::Exited(_, status)) => {
                println!("[parent] child exited with status {status}");
            }
            Ok(WaitStatus::Signaled(_, signal, core_dump)) => {
                println!("[parent] child was signaled with {signal}");
                println!("[parent] upon the signal the child generated a core dump: {core_dump}")
            }
            Ok(WaitStatus::Stopped(_, _)) => {
                println!("[parent] child was stopped, continuing it");
                thread::sleep(Duration::from_secs(random_sleep_time.into()));
                kill(child, Signal::SIGCONT).expect("Failed to send SIGCONT");
            }
            Ok(WaitStatus::Continued(_)) => {
                println!("[parent] child was continued");
            }
            Ok(_) => {
                println!("[parent] waiting for child to change state");
            }
            Err(err) => {
                eprintln!("[parent] error waiting for child: {err}");
                break;
            }
        }
    }
}
