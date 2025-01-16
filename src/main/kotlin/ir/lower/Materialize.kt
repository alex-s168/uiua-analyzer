package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constantArr
import me.alex_s168.uiua.ir.modifyPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.FnType

val argArrMat = modifyPass(
    "arg array materialize",
    Prims.Comp.ARG_ARR,
    {true}
) { put, newVar, a ->
    val data = args
    val ty = outs[0].type as ArrayType
    require(ty.length != null) {
        "arg array out type needs to have comptime length. typechecker fucked up?"
    }

    if (ty.inner is FnType)
        return@modifyPass

    val shape = constantArr(newVar, data.size.toDouble(), type = Types.size, put = put)

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prims.Comp.EMIT_ARR_ALLOC_P),
        mutableListOf(outs[0], shape)
    ))

    data.forEachIndexed { index, src ->
        val idc = constantArr(newVar, index.toDouble(), type = Types.size, put = put)
        put(IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prims.Comp.ARR_STORE),
            mutableListOf(outs[0], idc, src)
        ))
    }
}.parallelWithoutDeepCopy()
