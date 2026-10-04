#ifndef _ZK_UNISTD_H
#define _ZK_UNISTD_H
#include <sys/types.h>
pid_t fork(void);
int execvp(const char *file, char *const argv[]);
int unlink(const char *path);
int close(int fd);
_Noreturn void _exit(int code);
#endif
