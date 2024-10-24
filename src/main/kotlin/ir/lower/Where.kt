package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerWhere = lowerPass(
    "lower where",
    { it.instr is PrimitiveInstr && it.instr.id == Prim.WHERE }
) { put, newVar, a ->
    val arr = args[0]
    val arrTy = arr.type as ArrayType

    val amountOfSet = newVar().copy(type = Types.size)
    put(IrInstr(
        mutableListOf(amountOfSet),
        PrimitiveInstr(Prim.Comp.COUNT_NOTZERO),
        mutableListOf(arr)
    ))

    TODO()
}.parallelWithoutDeepCopy()