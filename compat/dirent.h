#pragma once
/*
 * Minimal dirent.h for Windows — provides opendir/readdir/closedir
 * using the Win32 FindFirstFile / FindNextFile / FindClose API.
 */

#ifdef _WIN32

#include <windows.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DT_UNKNOWN 0
#define DT_REG     1
#define DT_DIR     2

struct dirent {
    char   d_name[260];
    int    d_type;
};

typedef struct {
    HANDLE           handle;
    WIN32_FIND_DATAA find_data;
    struct dirent    entry;
    int              first;
} DIR;

static DIR* opendir(const char* path) {
    if (!path) return NULL;
    size_t plen = strlen(path);
    char   pattern[512];
    // Build "<path>\\*" search pattern
    if (plen > 0 && (path[plen - 1] == '/' || path[plen - 1] == '\\'))
        snprintf(pattern, sizeof(pattern), "%s*", path);
    else
        snprintf(pattern, sizeof(pattern), "%s\\*", path);

    DIR* dp = (DIR*)calloc(1, sizeof(DIR));
    if (!dp) return NULL;
    dp->handle = FindFirstFileA(pattern, &dp->find_data);
    if (dp->handle == INVALID_HANDLE_VALUE) {
        free(dp);
        return NULL;
    }
    dp->first = 1;
    return dp;
}

static struct dirent* readdir(DIR* dp) {
    if (!dp) return NULL;

    if (dp->first) {
        dp->first = 0;
    } else {
        if (!FindNextFileA(dp->handle, &dp->find_data))
            return NULL;
    }

    strncpy(dp->entry.d_name, dp->find_data.cFileName, sizeof(dp->entry.d_name) - 1);
    dp->entry.d_name[sizeof(dp->entry.d_name) - 1] = '\0';
    dp->entry.d_type = (dp->find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                           ? DT_DIR : DT_REG;
    return &dp->entry;
}

static int closedir(DIR* dp) {
    if (!dp) return -1;
    FindClose(dp->handle);
    free(dp);
    return 0;
}

#ifdef __cplusplus
}
#endif

#else
// On POSIX systems, use the real dirent.h
#include_next <dirent.h>
#endif
