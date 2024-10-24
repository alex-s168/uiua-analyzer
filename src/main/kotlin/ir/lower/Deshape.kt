package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerDeshape = lowerPrimPass(Prim.DESHAPE) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.RESHAPE_VIEW),
        mutableListOf(args[0])
    ))
}.parallelWithoutDeepCopy()