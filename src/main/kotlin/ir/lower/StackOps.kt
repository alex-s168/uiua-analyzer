package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerDup = lowerPrimPass(Prims.DUP) { put, newVar, a ->
    outs.forEach {
        put(IrInstr(
            mutableListOf(it),
            PrimitiveInstr(Prims.Comp.USE),
            mutableListOf(args[0]),
        ))
    }
}.parallelWithoutDeepCopy()

val lowerFlip = lowerPrimPass(Prims.FLIP) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.USE),
        mutableListOf(args[1]),
    ))
    put(IrInstr(
        mutableListOf(outs[1]),
        PrimitiveInstr(Prims.Comp.USE),
        mutableListOf(args[0]),
    ))
}.parallelWithoutDeepCopy()

val lowerOver = lowerPrimPass(Prims.OVER) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.USE),
        mutableListOf(args[1]),
    ))
    put(IrInstr(
        mutableListOf(outs[1]),
        PrimitiveInstr(Prims.Comp.USE),
        mutableListOf(args[0]),
    ))
    put(IrInstr(
        mutableListOf(outs[2]),
        PrimitiveInstr(Prims.Comp.USE),
        mutableListOf(args[1]),
    ))
}.parallelWithoutDeepCopy()