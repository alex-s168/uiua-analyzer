#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#define ZERO_ALLOC
#define CHECK_ALLOC
#define CHECK_RANK
//#define PRINT_DIM_ONLY

#define GenArrType(dim, of) \
    typedef struct { \
        void *alloc; \
        of *aligned; \
        size_t elemsOff;     \
        size_t sizes[dim];   \
        size_t strides[dim]; \
    } Arr##dim##_##of;

#define GenUnrankedArrType(of) \
    typedef struct { \
        size_t rank; \
        void *alloc; \
        of *aligned; \
        size_t elemsOff; \
        size_t *sizes;   \
        size_t *strides; \
    } Arru_##of;

#define Data(arr)  (arr.aligned+arr.elemsOff)

static void stridesFromSizes (size_t* stridesOut, const size_t* sizesIn, size_t rank) {
    for (size_t i = 0; i < rank-1; i ++) {
        stridesOut[i] = sizesIn[i+1];
    }
    stridesOut[rank-1] = 1;
}

static void checkAllocImpl (void* alloc) {
    if (alloc == NULL) {
        fprintf(stderr, "not enough memory for array!\n");
        exit(1);
    }
}

static void checkRankImpl (size_t actual, size_t want) {
    if (actual != want) {
        fprintf(stderr, "Incompatible ranks! %zu vs %zu (want %zu)\n", actual, want, want);
        exit(1);
    }
}

static void unshapedArrCopyImpl (char* dest, size_t strideDest,
                                 const char* src, size_t strideSrc,
                                 size_t numEl, size_t elSize) {
    for (size_t i = 0; i < numEl; i ++) {
        const char* a = src + strideSrc*elSize*i;
        char* b = dest + strideDest*elSize*i;
        memcpy(b, a, elSize);
    }
}

#define unshapedArrCopy(dst, dstRank, src, srcRank, numEl, elty) \
    unshapedArrCopyImpl((char*) (dst.aligned+dst.elemsOff), dst.strides[dstRank-1], \
                        (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-1], \
                        numEl, sizeof(elty))

#define deshapeIntoCArr(dst, src, srcRank, numEl, elty) \
    unshapedArrCopyImpl((char*) dst, 1, \
                        (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-1], \
                        numEl, sizeof(elty))

static void unshapedArrPickCopyImpl (char* dest, size_t strideDest,
                                     const char* src, size_t stride0src, size_t stride1src,
                                     size_t idx, size_t numEl, size_t elSize) {
    src += stride0src*elSize*idx;
    unshapedArrCopyImpl(dest, strideDest,
                        src, stride1src,
                        numEl, elSize);
}

// pick from innermost-1
#define unshapedArrPickRight2Copy(dst, dstRank, src, srcRank, idx, elty) \
    unshapedArrPickCopyImpl( \
            (char*) (dst.aligned+dst.elemsOff), dst.strides[dstRank-1], \
            (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-2], src.strides[srcRank-1], \
            idx, src.sizes[src.rank-1], sizeof(elty))

// pick from innermost-1
#define deshapeIntoCArrPickRight2(dst, src, srcRank, idx, elty) \
    unshapedArrPickCopyImpl( \
            (char*) dst, 1,  \
            (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-2], src.strides[srcRank-1], \
            idx, src.sizes[srcRank-1], sizeof(elty))

#define arrDealloc(arr) free(arr.alloc)

#ifdef CHECK_ALLOC
# define checkAlloc(a) checkAllocImpl(a);
#else 
# define checkAlloc(a) ; 
#endif

#ifdef ZERO_ALLOC
# define zeroAlloc(ptr, len) memset(ptr, 0, len);
#else 
# define zeroAlloc(ptr, len) ;
#endif

#ifdef CHECK_RANK 
# define checkRank(actual, want) checkRankImpl(actual, want);
#else 
# define checkRank(actual, want) ;
#endif 

#define GenUnrankedUtils(of) \
static size_t numUElements_##of (Arru_##of input) { \
    size_t res = 0; \
    for (size_t i = 0; i < input.rank; i ++) \
        res += input.sizes[i]; \
    return res; \
} \
\
static void fillUArr_##of (Arru_##of arr, of value) { \
    of* ptr = arr.aligned + arr.elemsOff; \
    size_t stride = arr.strides[arr.rank-1]; \
    size_t numEl = numUElements_##of(arr); \
    for (size_t i = 0; i < numEl; i ++) { \
        ptr[stride*i] = value; \
    } \
} \
\
static Arru_##of allocUArr_##of (size_t numEl, size_t rank, const size_t *sizesIn) { \
    Arru_##of res; \
    res.alloc = malloc(sizeof(of)*numEl + sizeof(size_t)*rank*2); \
    checkAlloc(res.alloc) \
    res.aligned = (of*) res.alloc; \
    res.elemsOff = 0; \
    res.rank = rank; \
    res.sizes = (size_t*) (res.alloc + sizeof(of)*numEl); \
    res.strides = res.sizes + rank; \
    \
    memcpy(res.sizes, sizesIn, sizeof(size_t)*rank); \
    stridesFromSizes(res.strides, sizesIn, rank); \
    \
    zeroAlloc(res.alloc, sizeof(of)*numEl); \
    \
    return res; \
} \
\
static of* cloneUToCArr_##of(size_t* numElOut, Arru_##of arr) { \
    size_t numEl = numUElements_##of(arr); \
    if (numElOut) *numElOut = numEl; \
    of* flat = malloc(sizeof(of)*numEl); \
    checkAlloc(flat); \
    deshapeIntoCArr(flat, \
                    arr, arr.rank, \
                    numEl, of); \
    return flat; \
}

#define GenRankedUtils(trank, of) \
static Arr##trank##_##of allocArr_##trank##of (size_t numEl, const size_t *sizesIn) { \
    Arr##trank##_##of res; \
    res.alloc = malloc(sizeof(of)*numEl); \
    checkAlloc(res.alloc); \
    res.aligned = (of*) res.alloc; \
    res.elemsOff = 0; \
    \
    memcpy(res.sizes, sizesIn, sizeof(size_t)*trank); \
    stridesFromSizes(res.strides, sizesIn, trank); \
    \
    zeroAlloc(res.alloc, sizeof(of)*numEl); \
    \
    return res; \
} \
\
static size_t numElements_##trank##of (Arr##trank##_##of input) { \
    size_t res = 0; \
    for (size_t i = 0; i < trank; i ++) \
        res += input.sizes[i]; \
    return res; \
} \
\
static Arru_##of cloneToUnranked_##trank##of (Arr##trank##_##of input) { \
    size_t numEl = numElements_##trank##of(input); \
    Arru_##of res = allocUArr_##of(numEl, trank, input.sizes); \
    unshapedArrCopy(res, trank, \
                    input, trank, \
                    numEl, of); \
    return res; \
} \
\
static Arr##trank##_##of cloneToRanked_##trank##of (Arru_##of input) { \
    size_t numEl = numUElements_##of(input); \
    checkRank(input.rank, trank); \
    Arr##trank##_##of res = allocArr_##trank##of(numEl, input.sizes); \
    unshapedArrCopy(res, trank, \
                    input, trank, \
                    numEl, of); \
    return res; \
} \
\
static of* cloneToCArr_##trank##of(size_t* numElOut, Arr##trank##_##of arr) { \
    size_t numEl = numElements_##trank##of(arr); \
    if (numElOut) *numElOut = numEl; \
    of* flat = malloc(sizeof(of)*numEl); \
    checkAlloc(flat); \
    deshapeIntoCArr(flat, \
                    arr, trank, \
                    numEl, of); \
    return flat; \
}

#define GenExtendScalar(ofTy, nameExt) \
extern Arr1_##ofTy _$_rt_extendScalar_##nameExt (Arr1_##ofTy base, size_t targetLen, ofTy fillWith) { \
    Arr1_##ofTy res; \
    res.alloc = calloc(targetLen, sizeof(ofTy)); \
    checkAlloc(res.alloc) \
    res.aligned = res.alloc; \
    res.elemsOff = 0; \
    res.sizes[0] = targetLen; \
    res.strides[0] = 1; \
    \
    size_t j = 0; \
    \
    ofTy *basePtr = base.aligned + base.elemsOff; \
    for (size_t i = 0; i < base.sizes[0]; i += base.strides[0]) { \
        res.aligned[j] = basePtr[i]; \
        j ++; \
    } \
    \
    for (; j < targetLen; j ++) { \
        res.aligned[j] = fillWith; \
    } \
    \
    return res; \
}

#define GenExtendRepeat(ofTy, nameExt) \
extern Arr1_##ofTy _$_rt_extendRepeat_##nameExt (Arr1_##ofTy base, size_t targetLen, Arr1_##ofTy fillWith) { \
    Arr1_##ofTy res; \
    res.alloc = calloc(targetLen, sizeof(ofTy)); \
    checkAlloc(res.alloc) \
    res.aligned = res.alloc; \
    res.elemsOff = 0; \
    res.sizes[0] = targetLen; \
    res.strides[0] = 1; \
    \
    size_t j = 0; \
    \
    ofTy *basePtr = base.aligned + base.elemsOff; \
    for (size_t i = 0; i < base.sizes[0]; i += base.strides[0]) { \
        res.aligned[j] = basePtr[i]; \
        j ++; \
    } \
    \
    ofTy *fillWithPtr = fillWith.aligned + fillWith.elemsOff; \
    size_t fillWithLen = fillWith.sizes[0]; \
    for (size_t x = 0; j < targetLen; j ++) { \
        res.aligned[j] = fillWithPtr[x % fillWithLen]; \
        x ++; \
    } \
    \
    return res; \
}

#define GenPrintDim(of, uacName) \
static void printUArrDim_##of(Arru_##of arr, FILE* stream, size_t* writtenOut) { \
    size_t written = 1; \
    fprintf(stream, "<"); \
    for (size_t i = 0; i < arr.rank; i ++) { \
        if (i > 0) { \
            fprintf(stream, " x "); \
            written += 3; \
        } \
        char buf[32]; \
        sprintf(buf, "%zu", arr.sizes[i]); \
        fputs(buf, stream); \
        written += strlen(buf); \
    } \
    fprintf(stream, " x " uacName ">\n"); \
    written += 4; \
    written += strlen(uacName); \
    if (writtenOut) *writtenOut = written; \
}

#ifdef PRINT_DIM_ONLY
# define GenPrint(of, uacName, customCode, printEl) \
static void printUArr_##of(Arru_##of arr, FILE* stream, size_t* writtenOut) { \
    printUArrDim_##of(arr, stream, writtenOut); \
}
#else 
# define GenPrint(of, uacName, customCode, printEl) \
static void printUArr_##of(Arru_##of arr, FILE* stream, size_t* writtenOut) { \
    if (arr.rank > 2) { \
        printUArrDim_##of(arr, stream, writtenOut); \
        return; \
    } \
    \
    customCode; \
    \
    if (arr.rank == 1) { \
        size_t written = 1; \
        fprintf(stream, "["); \
        of* elems = arr.aligned + arr.elemsOff; \
        for (size_t i = 0; i < arr.sizes[0]; i ++) { \
            if (i > 0) { \
                written += 2; \
                fprintf(stream, ", "); \
            } \
            of el = elems[arr.strides[0] * i]; \
            char buf[128]; \
            printEl; \
            fputs(buf, stream); \
            written + strlen(buf); \
        } \
        written += 1; \
        fprintf(stream, "]\n"); \
        if (writtenOut) *writtenOut = written; \
        return; \
    } \
    /* rank = 2 */ \
    \
    fputs("┌\n", stream); \
    size_t maxLen = 0; \
    for (size_t i = 0; i < arr.sizes[0]; i ++) { \
        Arru_##of inner = allocUArr_##of(arr.sizes[1], 1, &arr.sizes[1]); \
        unshapedArrPickRight2Copy(inner, inner.rank, arr, arr.rank, i, uint8_t); \
        size_t written; \
        printUArr_##of(inner, stream, &written); \
        arrDealloc(inner); \
        if (written > maxLen) maxLen = written; \
    } \
    for (size_t i = 0; i < maxLen+1; i ++) \
        fputc(' ', stream); \
    fputs("┘\n", stream); \
    \
    if (writtenOut) *writtenOut = 0; \
}
#endif

#define GenEverything(type, uacName) \
    GenArrType(1, type) \
    GenUnrankedArrType(type) \
    GenUnrankedUtils(type) \
    GenRankedUtils(1, type) \
    GenExtendScalar(type, uacName) \
    GenExtendRepeat(type, uacName)

GenEverything(uint8_t, byte)
GenEverything(int64_t, int)
GenEverything(double,  flt)
GenEverything(size_t,  size)

GenArrType(2, int64_t)
GenRankedUtils(2, int64_t)

typedef Arr1_uint8_t String;

static bool uarrIsString(Arru_uint8_t arr) {
    uint8_t* ptr = arr.aligned + arr.elemsOff;
    size_t stride = arr.strides[arr.rank-1];
    size_t numEl = numUElements_uint8_t(arr);
    for (size_t i = 0; i < numEl; i ++) {
        char c = (char) (ptr[i*stride]);
        if (c > 127) return false;
    }
    return true;
}

static bool arrIsString(Arr1_uint8_t arr) {
    uint8_t* ptr = arr.aligned + arr.elemsOff;
    for (size_t i = 0; i < arr.sizes[0]; i ++) {
        char c = (char) (ptr[i*arr.strides[0]]);
        if (c > 127) return false;
    }
    return true;
}

typedef struct {
    char* ptr;
    bool  shouldFree;
} String_toC_res;

static_assert(sizeof(uint8_t) == sizeof(char),
              "char size != 1 byte");

static String_toC_res String_toC(String str) {
    String_toC_res r;
    if (str.strides[0] == 1) {
        r.shouldFree = false;
        r.ptr = (char*) (str.aligned + str.elemsOff);
    } else {
        r.shouldFree = true;
        r.ptr = (char*) cloneToCArr_1uint8_t(NULL, str);
    } 

    return r;
}

static void String_print(String str, FILE* stream) {
    String_toC_res cstr = String_toC(str);
    fputs(cstr.ptr, stream);
    if (cstr.shouldFree) free(cstr.ptr);
}

static void String_println(String str, FILE* stream) {
    String_print(str, stream);
    fputc('\n', stream);
}

GenPrintDim(uint8_t, "byte")
GenPrint   (uint8_t, "byte", {
    if (arr.rank == 1) {
        if (uarrIsString(arr)) {
            String str = cloneToRanked_1uint8_t(arr);
            fputs("\"", stream);
            String_print(str, stream);
            fputs("\"\n", stream);
            arrDealloc(str);
            return;
        }
    }
    else { // rank 2
        if (uarrIsString(arr)) {
            fputs("┌\n", stream);

            char* inner = malloc(sizeof(char) * arr.sizes[1]);
            for (size_t i = 0; i < arr.sizes[0]; i ++) {
                deshapeIntoCArrPickRight2((uint8_t*) inner, arr, arr.rank, i, uint8_t);
                fprintf(stream, " \"%s\"\n", inner);
                fputs("\"\n", stream);
            }
            free(inner);

            size_t maxWidth = arr.sizes[1] + 3;
            for (size_t i = 0; i < maxWidth; i ++)
                fputc(' ', stream);
            fputs("┘\n", stream);
        }
    }
}, { sprintf(buf, "%b", el); })

GenPrintDim(int64_t, "int")
GenPrint   (int64_t, "int",  {}, { sprintf(buf, "%lld", el); })

GenPrintDim(double, "flt")
GenPrint   (double, "flt",  {}, { sprintf(buf, "%f", el); })

GenPrintDim(size_t, "size")
GenPrint   (size_t, "size", {}, { sprintf(buf, "%zu", el); })

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

extern char * debugInformation;

#define GetUasmHeaderp() ((UasmHeader*) (void*) debugInformation)

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

static void PrintSpan(size_t id) {
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

__attribute__ ((noreturn))
extern void _$_rt_panic(int64_t block, int64_t inst, int64_t dbgBefore, int64_t dbgAfter) {
    if (dbgBefore == -1)
        dbgBefore = dbgAfter;

    int64_t guessed = -1;
    bool inaccurate = false;
    if (dbgBefore == -1) {
        fprintf(stderr, "Runtime panic in compiled code without source location");
    }
    else if (dbgBefore == dbgAfter) {
        fprintf(stderr, "Runtime panic in compiled code in code span %llu", dbgBefore);
        guessed = dbgBefore;
    }
    else {
        fprintf(stderr, "Runtime panic in compiled code between code span %llu and code span %llu", dbgBefore, dbgAfter);
        guessed = dbgBefore;
        inaccurate = true;
    }

    fprintf(stderr, "! uiuac IR location: [%llu, %llu]\n", block, inst);

    if (inaccurate) {
        fprintf(stderr, "(warning: might be inaccurate!)\n");
    }
    PrintSpan(guessed);

    exit(1);
    for(;;);
}

// TODO: somehow tell mlir to use c struct arg

extern Arr2_int64_t fn_$__arr_start_arr_start_int_end__maybe__end__maybe_vaoff(Arr2_int64_t);

int main() {
    Arr2_int64_t a = allocArr_2int64_t(9, (size_t[]) {3,3});
    Data(a)[3* 0+0] = 1;
    Data(a)[3* 1+1] = 2;
    Data(a)[3* 2+2] = 3;

    Arr2_int64_t res = fn_$__arr_start_arr_start_int_end__maybe__end__maybe_vaoff(a);
    Arru_int64_t ures = cloneToUnranked_2int64_t(res);
    arrDealloc(res);
    printUArr_int64_t(ures, stdout, NULL);
    arrDealloc(ures);

    return 0;
}
