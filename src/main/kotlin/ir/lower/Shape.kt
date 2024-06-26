package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerShape = lowerPrimPass(Prim.SHAPE) { put, newVar, a ->
    val arr = args[0]
    val arrTy = arr.type as ArrayType

    val dims = List(arrTy.shape.size) { dim ->
        val const = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(const),
            NumImmInstr(dim.toDouble()),
            mutableListOf()
        ))

        val v = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(v),
            PrimitiveInstr(Prim.Comp.DIM),
            mutableListOf(arr, const)
        ))

        v
    }.wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_MATERIALIZE),
        mutableListOf(dims)
    ))
}