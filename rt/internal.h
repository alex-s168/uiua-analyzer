#include "rt.h"

#define ZERO_ALLOC
#define CHECK_ALLOC
#define CHECK_RANK
//#define PRINT_DIM_ONLY
#define HAVE_DEBUG_INFO
#define DYNCAST_CHECK
#define HAVE_INTERPRETER


static void checkAllocImpl (void* alloc) {
    if (alloc == NULL) {
        fprintf(stderr, "not enough memory for array!\n");
        uac_panic();
    }
}

static void checkRankImpl (size_t actual, size_t want) {
    if (actual != want) {
        fprintf(stderr, "Incompatible ranks! %zu vs %zu (want %zu)\n", actual, want, want);
        uac_panic();
    }
}

static void checkCastImpl (uac_TypeId actual, uac_TypeId want) {
    if (actual != want) {
        fprintf(stderr, "Can't cast %s to %s!\n",
                        uac_TypeId_name(actual),
                        uac_TypeId_name(want));
        uac_panic();
    }
}

extern OptInstSpan uac_currentSpan;

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

#ifdef DYNCAST_CHECK
# define checkCast(actual, want) checkCastImpl(actual, want);
#else 
# define checkCast(actual, want) ;
#endif

typedef struct {
    size_t len;
    void*  data;
} LightCArr;
