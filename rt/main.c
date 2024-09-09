#include <stdint.h>
#include "internal.h"


extern OptInstSpan uac_currentSpan = -1;

void uac_panic() {
    if (uac_currentSpan == -1) {
        fprintf(stderr, "Runtime panic in compiled code without source location!\n");

        uac_printSpan(uac_currentSpan);
    }
    else {
        fprintf(stderr, "Runtime panic in compiled code in code span %llu!\n", uac_currentSpan);
    }

    exit(1);
    for(;;);
}

void arrDeallocImpl(void* ptr) {
    if (ptr) { free(ptr); }
}

const char* uac_TypeId_name(uac_TypeId ty) {
    switch (ty) {
        case UAC_NOTYPE:
        default:
            return "???";

        case UAC_BYTE:
            return "byte";
        case UAC_INT:
            return "int";
        case UAC_FLT:
            return "flt";
        case UAC_SIZE:
            return "size";
        case UAC_DYN:
            return "dyn";

        case UAC_ARR_BYTE:
            return "rarr[byte]";
        case UAC_ARR_INT:
            return "rarr[int]";
        case UAC_ARR_FLT:
            return "rarr[flt]";
        case UAC_ARR_SIZE:
            return "rarr[size]";
        case UAC_ARR_DYN:
            return "rarr[dyn]";
    }
}

#define GenDynCastPrim(T, tyid) \
T uac_Dyn_as_##T(uac_Dyn dyn) { \
    checkCast(dyn.ty, tyid) \
    if (sizeof(T) <= sizeof(void*)) { \
        return (T) (intptr_t) dyn.opaque; \
    } \
    return * (T*) dyn.opaque; \
} \
\
uac_Dyn uac_Dyn_from_##T(T inp) { \
    if (sizeof(T) <= sizeof(void*)) { \
        return (uac_Dyn) { \
            .ty = tyid, \
            .opaque = (void*) (intptr_t) inp \
        }; \
    } \
    T* ptr = (T*) malloc(sizeof(T)); \
    if (ptr == NULL) { \
        return (uac_Dyn) { .ty = UAC_NOTYPE, .opaque = NULL }; \
    } \
    *ptr = inp; \
    return (uac_Dyn) { .ty = tyid, .opaque = (void*) ptr }; \
}

#define GenDynCastStruct(T, tyid) \
T uac_Dyn_as_##T(uac_Dyn dyn) { \
    checkCast(dyn.ty, tyid) \
    static_assert(sizeof(T) > sizeof(void*), "bruh"); \
    return * (T*) dyn.opaque; \
} \
\
uac_Dyn uac_Dyn_from_##T(T inp) { \
    static_assert(sizeof(T) > sizeof(void*), "bruh"); \
    T* ptr = (T*) malloc(sizeof(T)); \
    if (ptr == NULL) { \
        return (uac_Dyn) { .ty = UAC_NOTYPE, .opaque = NULL }; \
    } \
    *ptr = inp; \
    return (uac_Dyn) { .ty = tyid, .opaque = (void*) ptr }; \
}

GenDynCastPrim(uint8_t, UAC_BYTE);
GenDynCastStruct(Arru_uint8_t, UAC_ARR_BYTE);

GenDynCastPrim(int64_t, UAC_INT);
GenDynCastStruct(Arru_int64_t, UAC_ARR_INT);

GenDynCastPrim(double, UAC_FLT);
GenDynCastStruct(Arru_double, UAC_ARR_FLT);

GenDynCastPrim(size_t, UAC_SIZE);
GenDynCastStruct(Arru_size_t, UAC_ARR_SIZE);

GenDynCastStruct(uac_Dyn, UAC_DYN);
GenDynCastStruct(Arru_uac_Dyn, UAC_ARR_DYN);

void uac_Dyn_drop(uac_Dyn dyn) {
    switch (dyn.ty) {
#define G(ty, id) case id: if (sizeof(ty) <= sizeof(void*)) { return; } break;
        G(uint8_t, UAC_BYTE)
        G(int64_t, UAC_INT)
        G(double,  UAC_FLT)
        G(size_t,  UAC_SIZE)
#undef  G
        default: break;
    }

    if (dyn.opaque) free(dyn.opaque);
}

__attribute__ ((noreturn))
extern void _$_rt_panic(OptInstSpan span, int64_t block, int64_t inst) {
    uac_currentSpan = span;

    fprintf(stderr, "uiuac IR location: [%llu, %llu]\n", block, inst);
    uac_panic();
}

__attribute__ ((noreturn))
static void errNoInterpreter() {
    fprintf(stderr, "Uiua interpretation requested by compiled code but was not available!\n");
    uac_panic();
}

#ifdef HAVE_INTERPRETER

extern void uac_interpretImpl(const char* uasm_path,
                              size_t firstInstr, size_t lastInstr,
                              LightCArr /* <uac_Dyn> */ args,
                              LightCArr /* <uac_Dyn> */ rets); 

extern void _$_rt_interpret(OptInstSpan inst, size_t beginInstrId, size_t endInstrId,
                            const Arr1_uac_Dyn inputs, Arr1_uac_Dyn outputs) {

    if (uac_uasmFilePath == NULL) {
        errNoInterpreter();
    }

    LightCArr inputsa = (LightCArr) { .len = inputs.sizes[0], .data = Data(inputs) };
    LightCArr outputsa = (LightCArr) { .len = outputs.sizes[0], .data = Data(outputs) };

    uac_interpretImpl(uac_uasmFilePath, beginInstrId, endInstrId, inputsa, outputsa);
}

#else 

extern void _$_rt_interpret(OptInstSpan inst, size_t beginInstrId, size_t endInstrId,
                            const Arr1_uac_Dyn inputs, Arr1_uac_Dyn outputs) {
    errNoInterpreter();
}

#endif

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
