/*
    g++ -o fork-exec-shell.out fork-exec-shell.cpp
*/
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <string>
#include <vector>
#include <cstring>  // std:strerror.
#include <sys/types.h>  // mode_t used in safeOpen.
#include <fcntl.h>  // open.
#include <sstream>


class FileDescriptorException : public std::runtime_error {
    public:
        FileDescriptorException(const std::string& msg) : std::runtime_error(msg) {}
};

void safeClose(int fd, const std::string& description);
int safeOpen(const std::string& path, int flags, mode_t mode);
std::string argsToString(const std::vector<std::string>& args);
void safeExecvp(const std::vector<std::string>& args);


int main(int argc, char* argv[]) {
    pid_t pid = fork();
    if (pid < 0) {  // For failed.
        std::cerr << "fork failed" << std::endl;
        return -1;
    } else if (pid == 0) {  // The child process.
        std::cout << "[child] parent pid: " << getppid() << "\ncurrent pid: " << getpid() << std::endl;

        try {
            safeClose(STDOUT_FILENO, "stdout");
            int newStdout = safeOpen("./output.log", O_CREAT|O_WRONLY|O_TRUNC, S_IRUSR|S_IWUSR);
        } catch (const std::exception& e) {
            std::cerr << "failed re-routing output: " << e.what() << std::endl;
            return -1;
        }

        std::vector<std::string> args = {"wv", "fork-exec-shell.cpp"};
        std::vector<char*> c_args;
        c_args.reserve(args.size() + 1);
        for (const auto& arg : args) {
            // a static_cast can't add or remove const-ness, and c_str() return a `const char*`
            // when we only need a `char*`.
            c_args.push_back(const_cast<char*>(arg.c_str()));
        }
        c_args.push_back(nullptr);

        // Rely on the "lowest-numbered unused descriptor" rule.
        try {
            safeExecvp(args);
        } catch (const std::exception& e) {
            std::cerr << "failed to execute command: " << e.what() << std::endl;
            std::cerr << argsToString(args) << std::endl;
            return -1;
        }
    } else {  // pid > 0 and equtl to the child's PID.
        pid_t pidChild = wait(nullptr);
        std::cout << "[parent] parent pid: " << getppid() << "\nparent pid: " << getpid() << "\nchild pid:  " << pidChild << std::endl;
    }
    return 0;
}


void safeClose(int fd, const std::string& description) {
    if (close(fd) == -1) {
        throw FileDescriptorException("failed to close " + description + ": " + std::strerror(errno));
    }
}

int safeOpen(const std::string& path, int flags, mode_t mode) {
    int fd = open(path.c_str(), flags, mode);
    if (fd == -1) {
        throw FileDescriptorException("failed to open " + path + ": " + std::strerror(errno));
    }
    return fd;
}

std::string argsToString(const std::vector<std::string>& args) {
    std::ostringstream oss;
    oss << "{ ";
    for (size_t i=0; i<args.size(); i++) {
        oss << args[i] << " ";
    }
    oss << "}";
    return oss.str();
}

void safeExecvp(const std::vector<std::string>& args) {
    std::vector<char*> c_args;
    c_args.reserve(args.size() + 1);
    for (const auto& arg : args) {
        // a static_cast can't add or remove const-ness, and c_str() return a `const char*`
        // when we only need a `char*`.
        c_args.push_back(const_cast<char*>(arg.c_str()));
    }
    c_args.push_back(nullptr);

    if (execvp(c_args[0], c_args.data()) == -1) {
        throw std::runtime_error("execvp() failed: " + std::string(std::strerror(errno)));
    }
}