#pragma once

#ifdef _MSC_VER
// MSVC: no POSIX mkdir/strcasecmp — map to underscored versions
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#define strcasecmp         _stricmp
#endif

#ifdef __MINGW32__
// MinGW UCRT: mkdir only takes 1 arg, strcasecmp exists natively
#include <direct.h>
#undef  mkdir
#define mkdir(path, mode) _mkdir(path)
#endif
