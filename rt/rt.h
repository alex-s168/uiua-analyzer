#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>


#define HGenArrType(dim, of) \
    typedef struct { \
        void *alloc; \
        of *aligned; \
        size_t elemsOff;     \
        size_t sizes[dim];   \
        size_t strides[dim]; \
    } Arr##dim##_##of;

#define HGenUnrankedArrType(of) \
    typedef struct { \
        size_t rank; \
        void *alloc; \
        of *aligned; \
        size_t elemsOff; \
        size_t *sizes;   \
        size_t *strides; \
    } Arru_##of;

// inst span id ; can be -1 to indicate not present
typedef int64_t OptInstSpan;
#define OptInstSpan_NONE ((OptInstSpan) -1)

#include "rt_hidden.h"

#define Data(arr) (arr.aligned+arr.elemsOff)
#define arrDealloc(arr) arrDeallocImpl(arr.alloc)

void uac_stridesFromSizes (size_t* stridesOut, const size_t* sizesIn, size_t rank);

// reshape copy 
#define unshapedArrCopy(dst, dstRank, src, srcRank, numEl, elty) \
    uac_unshapedArrCopyImpl((char*) (dst.aligned+dst.elemsOff), dst.strides[dstRank-1], \
                            (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-1], \
                            numEl, sizeof(elty))

// deshape copy into c array 
#define deshapeIntoCArr(dst, src, srcRank, numEl, elty) \
    uac_unshapedArrCopyImpl((char*) dst, 1, \
                            (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-1], \
                            numEl, sizeof(elty))

// reshape pick 1d array from innermost-1 array
#define unshapedArrPickRight2Copy(dst, dstRank, src, srcRank, idx, elty) \
    uac_unshapedArrPickCopyImpl( \
            (char*) (dst.aligned+dst.elemsOff), dst.strides[dstRank-1], \
            (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-2], src.strides[srcRank-1], \
            idx, src.sizes[src.rank-1], sizeof(elty))

// pick 1d from innermost-1 array into c array
#define deshapeIntoCArrPickRight2(dst, src, srcRank, idx, elty) \
    uac_unshapedArrPickCopyImpl( \
            (char*) dst, 1,  \
            (const char*) (src.aligned+src.elemsOff), src.strides[srcRank-2], src.strides[srcRank-1], \
            idx, src.sizes[srcRank-1], sizeof(elty))

#   define HGenUnrankedUtils(of) \
\
/** product of all elements in shape */ \
size_t numUElements_##of (Arru_##of input); \
\
/** set all elements in array to value */ \
void fillUArr_##of (Arru_##of arr, of value); \
\
/** allocates a new array with the given total num of elements, rank and shape */ \
Arru_##of allocUArr_##of (size_t numEl, size_t rank, const size_t *sizesIn); \
\
/** allocates a copy of the given array, you need to pass the sum of all elements in shape as arg */ \
of* cloneUToCArr_##of(size_t* numElOut, Arru_##of arr);

#   define HGenRankedUtils(trank, of) \
\
/** allocates a new array with the given total num of elements, and shape */ \
Arr##trank##_##of allocArr_##trank##of (size_t numEl, const size_t *sizesIn); \
\
/** product of all elements in shape */ \
size_t numElements_##trank##of (Arr##trank##_##of input); \
\
/** allocates a copy of the given array */ \
Arru_##of cloneToUnranked_##trank##of (Arr##trank##_##of input); \
\
/** allocates a copy of the given array */ \
Arr##trank##_##of cloneToRanked_##trank##of (Arru_##of input); \
\
/** clones this array to a flat C array */ \
of* cloneToCArr_##trank##of(size_t* numElOut, Arr##trank##_##of arr);

#   define HGenExtendScalar(ofTy, nameExt) \
extern Arr1_##ofTy _$_rt_extendScalar_##nameExt (OptInstSpan inst, Arr1_##ofTy base, size_t targetLen, ofTy fillWith);

#   define HGenExtendRepeat(ofTy, nameExt) \
extern Arr1_##ofTy _$_rt_extendRepeat_##nameExt (OptInstSpan inst, Arr1_##ofTy base, size_t targetLen, Arr1_##ofTy fillWith);

#   define HGenPrintDim(of, uacName) \
void printUArrDim_##of(Arru_##of arr, FILE* stream, size_t* writtenOut);

#   define HGenPrint(of, uacName) \
void printVal_Arru_##of(Arru_##of arr, FILE* stream, size_t* writtenOut); \
void printVal_##of(of value, FILE* stream, size_t* writtenOut);


// don't forget to uac_Dyn_drop() after not needed anymore ; doesn't free inner arrays
typedef struct uac_Dyn uac_Dyn;


#define HGenEverything(type, uacName) \
    HGenArrType(1, type) \
    HGenUnrankedArrType(type) \
    HGenUnrankedUtils(type) \
    HGenRankedUtils(1, type) \
    HGenExtendScalar(type, uacName) \
    HGenExtendRepeat(type, uacName) \
    HGenPrint(type, # uacName) \
    HGenPrintDim(type, # uacName)

HGenEverything(uint8_t, byte)
HGenEverything(int64_t, int)
HGenEverything(double,  flt)
HGenEverything(size_t,  size)
HGenEverything(uac_Dyn, dyn)

#undef HGenEverything
#undef HGenUnrankedUtils
#undef HGenRankedUtils
#undef HGenExtendRepeat
#undef HGenExtendScalar
// don't undef HGenArrType and HGenUnrankedArrType


typedef Arr1_uint8_t uac_String;

bool uarrIsString(Arru_uint8_t arr);
bool arrIsString(Arr1_uint8_t arr);

typedef struct {
    char* ptr;
    bool  shouldFree;
} String_toC_res;

static_assert(sizeof(uint8_t) == sizeof(char),
              "char size != 1 byte");

String_toC_res uac_String_toC(uac_String str);
void uac_String_print(uac_String str, FILE* stream);
void uac_String_println(uac_String str, FILE* stream);

void uac_printSpan(size_t id);

__attribute__ ((noreturn))
extern void _$_rt_panic(OptInstSpan at,
                        int64_t block, int64_t inst); // uac block and inst id ; can be -1 to indicate not present

typedef enum : uint8_t {
    UAC_NOTYPE = 0,
    UAC_BYTE,
    UAC_INT,
    UAC_FLT,
    UAC_SIZE,
    UAC_DYN,
    UAC_ARR_BYTE,
    UAC_ARR_INT,
    UAC_ARR_FLT,
    UAC_ARR_SIZE,
    UAC_ARR_DYN,  // identical to array of boxes
} uac_TypeId;

const char* uac_TypeId_name(uac_TypeId);

// don't forget to uac_Dyn_drop() after not needed anymore ; doesn't free inner arrays
// { i8, ptr }
struct uac_Dyn {
    uac_TypeId ty;
    void* opaque; // if sizeof(type) <= sizeof(void*) then bitcast and get your value ; otherwise deref
                  // you should use uac_Dyn_as_<T> or uac_Dyn_from_<T> instead of manually doing this 
};

void uac_Dyn_drop(uac_Dyn);

#define HGenDynCastI(T) \
    T uac_Dyn_as_##T(uac_Dyn dyn); \
    uac_Dyn uac_Dyn_from_##T(T inp);

#define HGenDynCast(T) \
    HGenDynCastI(T) \
    HGenDynCastI(Arru_##T)

HGenDynCast(uint8_t)
HGenDynCast(int64_t)
HGenDynCast(double)
HGenDynCast(size_t)
HGenDynCast(uac_Dyn)

#undef HGenDyncast
#undef HGenDynCastI

__attribute__ ((noreturn))
void uac_panic();

#define DynamicDispatch(dyn, fnPrefix, outCast, ...) \
switch ((dyn).ty) { \
    case UAC_NOTYPE: \
    default: \
        fprintf(stderr, "Invalid dyn[] type!\n"); \
        uac_panic(); \
\
    case UAC_BYTE: { \
        uint8_t outCast = uac_Dyn_as_uint8_t(dyn); \
        return fnPrefix##_uint8_t(__VA_ARGS__); } \
\
    case UAC_INT: { \
        int64_t outCast = uac_Dyn_as_int64_t(dyn); \
        return fnPrefix##_int64_t(__VA_ARGS__); } \
\
    case UAC_FLT: { \
        double outCast = uac_Dyn_as_double(dyn); \
        return fnPrefix##_double(__VA_ARGS__); } \
\
    case UAC_SIZE: { \
        size_t outCast = uac_Dyn_as_size_t(dyn); \
        return fnPrefix##_size_t(__VA_ARGS__); } \
\
    case UAC_DYN: { \
        uac_Dyn outCast = uac_Dyn_as_uac_Dyn(dyn); \
        return fnPrefix##_uac_Dyn(__VA_ARGS__); } \
\
    case UAC_ARR_BYTE: { \
        Arru_uint8_t outCast = uac_Dyn_as_Arru_uint8_t(dyn); \
        return fnPrefix##_Arru_uint8_t(__VA_ARGS__); } \
\
    case UAC_ARR_INT: { \
        Arru_int64_t outCast = uac_Dyn_as_Arru_int64_t(dyn); \
        return fnPrefix##_Arru_int64_t(__VA_ARGS__); } \
\
    case UAC_ARR_FLT: { \
        Arru_double outCast = uac_Dyn_as_Arru_double(dyn); \
        return fnPrefix##_Arru_double(__VA_ARGS__); } \
\
    case UAC_ARR_SIZE: { \
        Arru_size_t outCast = uac_Dyn_as_Arru_size_t(dyn); \
        return fnPrefix##_Arru_size_t(__VA_ARGS__); } \
\
    case UAC_ARR_DYN: { \
        Arru_uac_Dyn outCast = uac_Dyn_as_Arru_uac_Dyn(dyn); \
        return fnPrefix##_Arru_uac_Dyn(__VA_ARGS__); } \
}

extern const char * uac_uasmFilePath; // nullable

// interpret a range of instructions from the source UASM file ; only going to work if runtime configured correctly and uac_uasmFilePath is set
extern void _$_rt_interpret(OptInstSpan i, size_t beginInstrId, size_t endInstrId,
                            const Arr1_uac_Dyn inputs, Arr1_uac_Dyn outputs);
