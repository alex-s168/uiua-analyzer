package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.reduce

val lowerUnShape = lowerPrimPass<(IrBlock) -> Unit>(Prim.UN_SHAPE) { put, newVar, a, putBlock ->
    val shape = args[0]

    val mult = reduce(shape, put, putBlock, a.block.ref, newVar, Types.size) { a, b, res ->
        instrs += IrInstr(
            mutableListOf(res),
            PrimitiveInstr(Prim.MUL),
            mutableListOf(a, b)
        )
    }

    val rang = newVar().copy(type = Types.array(Types.int))
    put(IrInstr(
        mutableListOf(rang),
        PrimitiveInstr(Prim.RANGE),
        mutableListOf(mult)
    ))

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.RESHAPE),
        mutableListOf(shape, rang)
    ))
}.parallelWithoutDeepCopy()