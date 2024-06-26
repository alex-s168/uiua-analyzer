package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.into
import me.alex_s168.uiua.ir.optAwayPass
import me.alex_s168.uiua.ir.transform.unfailure

val comptimeReduceEval = optAwayPass(
    "comptime reduce eval",
    Prim.REDUCE,
    { a -> unfailure {
        val arr = args[1]
        val orig = a.origin(arr)!!
        val what = orig.args.first()
        a.origin(what)!!
        a.isPrim(orig, Prim.Comp.ARR_MATERIALIZE)
    } }
) { put, newVar, a ->
    val accTy = outs[0].type
    val arr = args[1]
    val first = args[0]
    val extra = args.subList(2, args.size - 1)
    val every = args.last()

    val arrSource = a.origin(arr)!!

    val what = arrSource.args.first()
    val argArr = a.origin(what)!!.args

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
        PrimitiveInstr(Prim.CALL),
        mutableListOf(first, elems[0], elems[1]).also { it += extra }
    ))

    elems.drop(2).fold(accInit) { acc, x ->
        val acc2 = newVar().copy(type = accTy)
        put(IrInstr(
            mutableListOf(acc2),
            PrimitiveInstr(Prim.CALL),
            mutableListOf(every, acc, x).also { it += extra }
        ))
        acc2
    }.into(outs[0], put)
}