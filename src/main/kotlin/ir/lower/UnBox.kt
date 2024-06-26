package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass

val lowerUnBox = lowerPrimPass(Prim.UN_BOX) { put, newVar, a ->
    put(IrInstr(
        outs,
        PrimitiveInstr(Prim.Comp.BOX_LOAD),
        args
    ))
}