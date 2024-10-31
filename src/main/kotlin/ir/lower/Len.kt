package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerLen = lowerPrimPass(Prims.LEN) { put, newVar, a ->
    if (args[0].type is ArrayType) {
        val const = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(const),
            NumImmInstr(0.0),
            mutableListOf()
        ))

        put(IrInstr(
            outs,
            PrimitiveInstr(Prims.Comp.DIM),
            mutableListOf(args[0], const)
        ))
    }
    else {
        put(IrInstr(
            mutableListOf(outs[0]),
            NumImmInstr(1.0),
            mutableListOf()
        ))
    }
}.parallelWithoutDeepCopy()