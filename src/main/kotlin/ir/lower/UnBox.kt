package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerUnBox = lowerPrimPass(Prims.UN_BOX) { put, newVar, a ->
    put(IrInstr(
        outs,
        PrimitiveInstr(Prims.Comp.BOX_LOAD),
        args
    ))
}.parallelWithoutDeepCopy()