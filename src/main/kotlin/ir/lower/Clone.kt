package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerClone = lowerPrimPass(Prim.Comp.ARR_CLONE) { put, newVar, a ->
    val arr = args[0]
    val arrTy = arr.type as ArrayType

    val dims = newVar().copy(type = Types.array(Types.size, arrTy.shape.size))
    put(IrInstr(
        mutableListOf(dims),
        PrimitiveInstr(Prim.SHAPE),
        mutableListOf(arr),
    ))

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
        mutableListOf(dims)
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.ARR_COPY),
        mutableListOf(outs[0], arr)
    ))
}.parallelWithoutDeepCopy()