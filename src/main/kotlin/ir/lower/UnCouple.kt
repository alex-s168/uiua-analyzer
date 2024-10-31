package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.constants

val lowerUnCouple = lowerPrimPass(Prims.UN_COUPLE) { put, newVar, a ->
    // TODO: runtime assert len == 2

    val (c0, c1) = constants(newVar, 0.0, 1.0, type = Types.size, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.PICK),
        mutableListOf(c0, args[0])
    ))

    put(IrInstr(
        mutableListOf(outs[1]),
        PrimitiveInstr(Prims.PICK),
        mutableListOf(c1, args[0])
    ))
}.parallelWithoutDeepCopy()