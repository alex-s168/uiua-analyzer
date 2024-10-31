package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerFix = lowerPrimPass(Prims.FIX) { put, putVar, a ->
    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.BOX),
        mutableListOf(args[0])
    ))
}.parallelWithoutDeepCopy()