#ifndef _ZK_SYS_STAT_H
#define _ZK_SYS_STAT_H
#include <sys/types.h>
struct stat { off_t st_size; time_t st_mtime; mode_t st_mode; };
int stat(const char *path, struct stat *st);
#endif
