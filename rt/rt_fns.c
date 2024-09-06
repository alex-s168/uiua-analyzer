#include "internal.h"

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

#define GenEverything(ty, uacName) \
    GenExtendScalar(ty, uacName) \
    GenExtendRepeat(ty, uacName)

GenEverything(uint8_t, byte)
GenEverything(int64_t, int)
GenEverything(double,  flt)
GenEverything(size_t,  size)


