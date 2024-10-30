package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerTranspose = lowerPrimPass(Prim.TRANSPOSE) { put, newVar, a ->
    val arrTy = args[0].type as ArrayType

    val sha = List(arrTy.shape.size) {
        val (d) = constants(newVar, it.toDouble(), type = Types.size, put = put)

        val dim = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(dim),
            PrimitiveInstr(Prim.Comp.DIM),
            mutableListOf(args[0], d)
        ))
        dim
    }.wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
        mutableListOf(sha)
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.TRANSPOSE),
        mutableListOf(outs[0], args[0])
    ))
}