void uac_unshapedArrCopyImpl (char* dest, size_t strideDest,
                              const char* src, size_t strideSrc,
                              size_t numEl, size_t elSize);

void uac_unshapedArrPickCopyImpl (char* dest, size_t strideDest,
                                  const char* src, size_t stride0src, size_t stride1src,
                                  size_t idx, size_t numEl, size_t elSize);

void arrDeallocImpl(void*);
