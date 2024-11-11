func.func private @_$_rt_primes(%num: i64) -> memref<? x i64, strided<[1]>>
func.func private @_$_rt_panic(%opt_span: i64, %block: i64, %instr: i64) {
    return
}
func.func private @_$_rt_time() -> f64
func.func private @_$_rt_extendRepeat_int(%span: i64, %base: memref<? x i64, strided<[1], offset: ?>>, %len: index, %fillWith: memref<? x i64, strided<[1], offset: ?>>) -> memref<? x i64, strided<[1], offset: ?>>
