package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.CallerInstrsCache
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.into

val comptimeReduceEval = optAwayPass(
    "comptime reduce eval",
    Prims.REDUCE,
    {true}
) { put, newVar, a ->
    val accTy = outs[0].type
    val arr = args[1]
    val first = args[0]
    val extra = args.subList(2, args.size - 1)
    val every = args.last()

    val cache = CallerInstrsCache()
    val argArr = a.argArrAsVars(arr, put, newVar, cache::get)
    if (argArr == null) {
        put(this)
        return@optAwayPass
    }

    val elems = if (argArr.size < 2) {
        val default = a.block.fillArg
            ?: error("(comptime catched) panic: no default for reduce with arg array")
        val new = argArr.toMutableList()
        repeat(2 - argArr.size) {
            new += default
        }
        new
    } else argArr

    val accInit = newVar().copy(type = accTy)
    put(IrInstr(
        mutableListOf(accInit),
        PrimitiveInstr(Prims.CALL),
        mutableListOf(first, elems[0], elems[1]).also { it += extra }
    ))

    elems.drop(2).fold(accInit) { acc, x ->
        val acc2 = newVar().copy(type = accTy)
        put(IrInstr(
            mutableListOf(acc2),
            PrimitiveInstr(Prims.CALL),
            mutableListOf(every, acc, x).also { it += extra }
        ))
        acc2
    }.into(outs[0], put)
}.parallelWithoutDeepCopy()