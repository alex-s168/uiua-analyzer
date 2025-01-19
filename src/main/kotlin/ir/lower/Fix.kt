package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerFix = lowerPrimPass(Prims.FIX) { put, newVar, a ->
    val arrTy = args[0].type

    if (arrTy is ArrayType) {
        val sha = List(arrTy.shape.size) {
            val (d) = constants(newVar, it.toDouble(), type = Types.size, put = put)

            val dim = newVar().copy(type = Types.size)
            put(IrInstr(
                mutableListOf(dim),
                PrimitiveInstr(Prims.Comp.DIM),
                mutableListOf(args[0], d)
            ))
            dim
        }

        put(IrInstr(
            mutableListOf(outs[0]),
            PrimitiveInstr(Prims.Comp.FIX_ARR),
            mutableListOf(args[0]).also { it += sha }
        ))
    } else {
        put(IrInstr(
            mutableListOf(outs[0]),
            PrimitiveInstr(Prims.BOX),
            mutableListOf(args[0])
        ))
    }
}.parallelWithoutDeepCopy()
