package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerBox = lowerPrimPass(Prim.BOX) { put, newVar, a ->
    put(IrInstr(
        outs,
        PrimitiveInstr(Prim.Comp.BOX_CREATE),
        mutableListOf()
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.BOX_STORE),
        mutableListOf(outs[0], args[0])
    ))
}.parallelWithoutDeepCopy()