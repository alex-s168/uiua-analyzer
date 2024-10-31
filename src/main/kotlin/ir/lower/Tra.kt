package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerTranspose = lowerPrimPass(Prims.TRANSPOSE) { put, newVar, a ->
    val arrTy = args[0].type as ArrayType

    val sha = List(arrTy.shape.size) {
        val (d) = constants(newVar, it.toDouble(), type = Types.size, put = put)

        val dim = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(dim),
            PrimitiveInstr(Prims.Comp.DIM),
            mutableListOf(args[0], d)
        ))
        dim
    }.wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.ARR_ALLOC),
        mutableListOf(sha)
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prims.Comp.TRANSPOSE),
        mutableListOf(outs[0], args[0])
    ))
}