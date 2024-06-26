package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass

val lowerDup = lowerPrimPass(Prim.DUP) { put, newVar, a ->
    outs.forEach {
        put(IrInstr(
            mutableListOf(it),
            PrimitiveInstr(Prim.Comp.USE),
            mutableListOf(args[0]),
        ))
    }
}

val lowerFlip = lowerPrimPass(Prim.FLIP) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.USE),
        mutableListOf(args[1]),
    ))
    put(IrInstr(
        mutableListOf(outs[1]),
        PrimitiveInstr(Prim.Comp.USE),
        mutableListOf(args[0]),
    ))
}

val lowerOver = lowerPrimPass(Prim.OVER) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.USE),
        mutableListOf(args[1]),
    ))
    put(IrInstr(
        mutableListOf(outs[1]),
        PrimitiveInstr(Prim.Comp.USE),
        mutableListOf(args[0]),
    ))
    put(IrInstr(
        mutableListOf(outs[2]),
        PrimitiveInstr(Prim.Comp.USE),
        mutableListOf(args[1]),
    ))
}