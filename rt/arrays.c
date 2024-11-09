#include "internal.h"

void uac_stridesFromSizes (size_t* stridesOut, const size_t* sizesIn, size_t rank) {
    for (size_t i = 0; i < rank-1; i ++) {
        stridesOut[i] = sizesIn[i+1];
    }
    stridesOut[rank-1] = 1;
}

void uac_unshapedArrCopyImpl (char* dest, size_t strideDest,
                              const char* src, size_t strideSrc,
                              size_t numEl, size_t elSize) {
    for (size_t i = 0; i < numEl; i ++) {
        const char* a = src + strideSrc*elSize*i;
        char* b = dest + strideDest*elSize*i;
        memcpy(b, a, elSize);
    }
}

void uac_unshapedArrPickCopyImpl (char* dest, size_t strideDest,
                                         const char* src, size_t stride0src, size_t stride1src,
                                         size_t idx, size_t numEl, size_t elSize) {
    src += stride0src*elSize*idx;
    uac_unshapedArrCopyImpl(dest, strideDest,
                            src, stride1src,
                            numEl, elSize);
}

#define GenUnrankedUtils(of) \
size_t numUElements_##of (Arru_##of input) { \
    size_t res = 1; \
    for (size_t i = 0; i < input.rank; i ++) \
        res *= input.sizes[i]; \
    return res; \
} \
\
void fillUArr_##of (Arru_##of arr, of value) { \
    of* ptr = arr.aligned + arr.elemsOff; \
    size_t stride = arr.strides[arr.rank-1]; \
    size_t numEl = numUElements_##of(arr); \
    for (size_t i = 0; i < numEl; i ++) { \
        ptr[stride*i] = value; \
    } \
} \
\
Arru_##of allocUArr_##of (size_t numEl, size_t rank, const size_t *sizesIn) { \
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
    uac_stridesFromSizes(res.strides, sizesIn, rank); \
    \
    zeroAlloc(res.alloc, sizeof(of)*numEl); \
    \
    return res; \
} \
\
of* cloneUToCArr_##of(size_t* numElOut, Arru_##of arr) { \
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
Arr##trank##_##of allocArr_##trank##of (size_t numEl, const size_t *sizesIn) { \
    Arr##trank##_##of res; \
    res.alloc = malloc(sizeof(of)*numEl); \
    checkAlloc(res.alloc); \
    res.aligned = (of*) res.alloc; \
    res.elemsOff = 0; \
    \
    memcpy(res.sizes, sizesIn, sizeof(size_t)*trank); \
    uac_stridesFromSizes(res.strides, sizesIn, trank); \
    \
    zeroAlloc(res.alloc, sizeof(of)*numEl); \
    \
    return res; \
} \
\
size_t numElements_##trank##of (Arr##trank##_##of input) { \
    size_t res = 1; \
    for (size_t i = 0; i < trank; i ++) \
        res *= input.sizes[i]; \
    return res; \
} \
\
Arru_##of cloneToUnranked_##trank##of (Arr##trank##_##of input) { \
    size_t numEl = numElements_##trank##of(input); \
    Arru_##of res = allocUArr_##of(numEl, trank, input.sizes); \
    unshapedArrCopy(res, trank, \
                    input, trank, \
                    numEl, of); \
    return res; \
} \
\
Arr##trank##_##of cloneToRanked_##trank##of (Arru_##of input) { \
    size_t numEl = numUElements_##of(input); \
    checkRank(input.rank, trank); \
    Arr##trank##_##of res = allocArr_##trank##of(numEl, input.sizes); \
    unshapedArrCopy(res, trank, \
                    input, trank, \
                    numEl, of); \
    return res; \
} \
\
of* cloneToCArr_##trank##of(size_t* numElOut, Arr##trank##_##of arr) { \
    size_t numEl = numElements_##trank##of(arr); \
    if (numElOut) *numElOut = numEl; \
    of* flat = malloc(sizeof(of)*numEl); \
    checkAlloc(flat); \
    deshapeIntoCArr(flat, \
                    arr, trank, \
                    numEl, of); \
    return flat; \
}

#define GenEverything(type, uacName) \
    GenUnrankedUtils(type) \
    GenRankedUtils(1, type) \

GenEverything(uint8_t, byte)
GenEverything(int64_t, int)
GenEverything(double,  flt)
GenEverything(size_t,  size)
GenEverything(uac_Dyn, uac_dyn)
