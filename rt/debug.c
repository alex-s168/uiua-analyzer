#include "internal.h"

#ifdef HAVE_DEBUG_INFO

typedef struct {
    size_t line;
    size_t col;
    size_t bytePos;
    size_t charPos;
} __attribute__((packed)) UasmLoc;

typedef struct {
    size_t  sourceFileId;
    UasmLoc begin;
    UasmLoc end;
} __attribute__((packed)) UasmSpan;

typedef struct {
    size_t fileNameByteLen; // with nt
    size_t sourceByteLen;   // with nt
} __attribute__((packed)) UasmSource;

typedef struct {
    size_t absOffsetOfSpans;
    size_t sourcesLen;
} __attribute__((packed)) UasmHeader;

extern char * uac_debugInformation;

#define GetUasmHeaderp() ((UasmHeader*) (void*) uac_debugInformation)

static const char * GetUasmSourceFileName(UasmSource* source) {
    char* bptr = ((char*) (void*) source) + sizeof(UasmSource);
    return bptr;
}

static const char * GetUasmSourceContents(UasmSource* source) {
    char* bptr = ((char*) (void*) source) + sizeof(UasmSource);
    return bptr + source->fileNameByteLen;
}

static UasmSource* GetUasmSource(size_t id) {
    UasmHeader* header = GetUasmHeaderp();
    if (id >= header->sourcesLen)
        return NULL;

    char* ptr = ((char*) (void*) header) + sizeof(UasmHeader);
    for (size_t i = 0; i < id; i ++) {
        UasmSource* src = (UasmSource*) (void*) ptr;
        ptr += sizeof(UasmSource);
        ptr += src->fileNameByteLen;
        ptr += src->sourceByteLen;
    }

    UasmSource* src = (UasmSource*) (void*) ptr;

    return src;
}

static UasmSpan* GetUasmSpan(size_t id) {
    UasmHeader* header = GetUasmHeaderp();
    UasmSpan* spansBegin = (UasmSpan*) (void*) (((char*) (void*) header) + header->absOffsetOfSpans);
    return &spansBegin[id];
}

static size_t getLineBreak(const char * str, size_t id) {
    size_t off = 0;
    for (; str[off]; off ++) {
        if (str[off] == '\n') {
            if (id == 0) return off;
            id --;
        }
    }
    return off;
}

void uac_printSpan(size_t id) {
    UasmSpan* span = GetUasmSpan(id);
    UasmSource* src = GetUasmSource(span->sourceFileId);
    const char * fileName = GetUasmSourceFileName(src);
    const char * source = GetUasmSourceContents(src);

    fprintf(stderr, "In file %s (line: %zu, column: %zu):\n",
                    fileName, span->begin.line, span->begin.col);

    size_t begin = getLineBreak(source, span->begin.line);
    size_t len = getLineBreak(source + begin, span->end.line - span->begin.line);

    size_t i;
    for (i = begin; i < span->begin.bytePos; i ++) {
        fputc(source[i], stderr);
    }
    fputs("\033[31;1;4m", stderr);
    for (; i < begin + span->end.bytePos; i ++) {
        fputc(source[i], stderr);
    }
    fputs("[0m", stderr);
    for (; i < begin + len; i ++) {
        fputc(source[i], stderr);
    }
}

#else 

void uac_printSpan(size_t id) {
    fprintf(stderr, "In UASM span %zu (program was compiled without debug information)\n", id);
}

#endif
