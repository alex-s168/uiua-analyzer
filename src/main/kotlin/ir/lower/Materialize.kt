package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constantArr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerMaterialize = lowerPrimPass(Prims.Comp.ARR_MATERIALIZE) { put, newVar, a ->
    val data = a.origin(args[0])!!.args

    val shape = constantArr(newVar, data.size.toDouble(), type = Types.size, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.ARR_ALLOC),
        mutableListOf(shape)
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