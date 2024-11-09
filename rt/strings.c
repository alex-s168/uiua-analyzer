#include "internal.h"

#define GenPrintDim(of, uacName) \
void printUArrDim_##of(Arru_##of arr, FILE* stream, size_t* writtenOut) { \
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
# define GenPrint(of, uacName, customCode) \
void printUArr_##of(Arru_##of arr, FILE* stream, size_t* writtenOut) { \
    printUArrDim_##of(arr, stream, writtenOut); \
}
#else 
# define GenPrint(of, uacName, customCode) \
void printVal_Arru_##of(Arru_##of arr, FILE* stream, size_t* writtenOut) { \
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
            size_t t; \
            printVal_##of(el, stream, &t); \
            written += t; \
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
        printVal_Arru_##of(inner, stream, &written); \
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

#define GenPrintScalar(of, printEl) \
void printVal_##of(of el, FILE* stream, size_t* writtenOut) { \
    char buf[128]; \
    printEl; \
    fputs(buf, stream); \
    if (writtenOut) *writtenOut = strlen(buf); \
}

GenPrintDim(uint8_t, "byte")
GenPrint   (uint8_t, "byte", {
    if (arr.rank == 1) {
        if (uarrIsString(arr)) {
            uac_String str = cloneToRanked_1uint8_t(arr);
            fputs("\"", stream);
            uac_String_print(str, stream);
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
})
GenPrintScalar(uint8_t, { sprintf(buf, "%b", el); })

GenPrintDim(int64_t, "int")
GenPrint   (int64_t, "int",  {})
GenPrintScalar(int64_t, { sprintf(buf, "%ld", el); })

GenPrintDim(double, "flt")
GenPrint   (double, "flt",  {})
GenPrintScalar(double, { sprintf(buf, "%f", el); })

GenPrintDim(size_t, "size")
GenPrint   (size_t, "size", {})
GenPrintScalar(size_t, { sprintf(buf, "%zu", el); })


GenPrintDim(uac_Dyn, "dyn")
GenPrint   (uac_Dyn, "dyn", {})
void printVal_uac_Dyn(uac_Dyn dyn, FILE* stream, size_t* writtenOut) {
    size_t written = 5;

    fputs("dyn[", stream);

    size_t t;
    DynamicDispatch(dyn, printVal, cast,
                    cast, stream, &t)
    written += t;

    fputc(']', stream);

    if (writtenOut) *writtenOut = written;
}


bool uarrIsString(Arru_uint8_t arr) {
    uint8_t* ptr = arr.aligned + arr.elemsOff;
    size_t stride = arr.strides[arr.rank-1];
    size_t numEl = numUElements_uint8_t(arr);
    for (size_t i = 0; i < numEl; i ++) {
        char c = (char) (ptr[i*stride]);
        if (c > 127) return false;
    }
    return true;
}

bool arrIsString(Arr1_uint8_t arr) {
    uint8_t* ptr = arr.aligned + arr.elemsOff;
    for (size_t i = 0; i < arr.sizes[0]; i ++) {
        char c = (char) (ptr[i*arr.strides[0]]);
        if (c > 127) return false;
    }
    return true;
}

String_toC_res uac_String_toC(uac_String str) {
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

void uac_String_print(uac_String str, FILE* stream) {
    String_toC_res cstr = uac_String_toC(str);
    fputs(cstr.ptr, stream);
    if (cstr.shouldFree) free(cstr.ptr);
}

void uac_String_println(uac_String str, FILE* stream) {
    uac_String_print(str, stream);
    fputc('\n', stream);
}
