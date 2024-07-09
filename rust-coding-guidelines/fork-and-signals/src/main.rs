use nix::sys::signal::{kill, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{fork, getpid, getppid, ForkResult, Pid};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::thread;
use std::time::Duration;

fn main() {
    let random_sleep_time: u32 = thread_rng().gen_range(0..=10);
    println!(
        "[main] child thread will sleep for {:.3}",
        random_sleep_time
    );

    match unsafe { fork() } {
        Ok(ForkResult::Parent { child, .. }) => {
            let parent_pid = Pid::parent();
            println!("[parent] parent pid: {parent_pid}");
            let current_pid = Pid::this();
            println!("[parent] current pid: {current_pid}");
            println!("[parent] child'd pid: {child}");

            let base: u32 = 6;
            match random_sleep_time.cmp(&base) {
                Ordering::Less => match kill(child, Signal::SIGSTOP) {
                    Ok(_) => println!("[parent] sent stop signal to child"),
                    Err(error) => eprintln!("[parent] error sending stop signal to child: {error}"),
                },
                _ => println!("[parent] child was not sent a signal"),
            }

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
                        println!(
                            "[parent] upon the signal the child generated a core dump: {core_dump}"
                        )
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
        Ok(ForkResult::Child) => {
            let parent_pid = getppid();
            println!("[child] parent pid: {parent_pid}");
            let pid = getpid();
            println!("[child] current pid: {pid}");

            println!("[child] going to sleep for {random_sleep_time} seconds");
            thread::sleep(Duration::from_secs(random_sleep_time.into()));
            println!("[child] waking up now");
        }
        Err(error) => eprintln!("error in fork(): {error}"),
    }
}
