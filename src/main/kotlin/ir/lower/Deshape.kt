package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.reduce
import me.alex_s168.uiua.ir.transform.wrapInArgArray

// TODO: NEEDS ARRAY COPYPYYY
val lowerDeshape = lowerPrimPass<(IrBlock) -> Unit>(Prims.DESHAPE) { put, newVar, a, putBlock ->
    val shape = newVar().copy(type = Types.array(Types.size))
    put(IrInstr(
        mutableListOf(shape),
        PrimitiveInstr(Prims.SHAPE),
        mutableListOf(args[0])
    ))

    val shapeMul = reduce(shape, put, putBlock, a.block.ref, newVar, Types.size) { a, b, res ->
        instrs += IrInstr(
            mutableListOf(res),
            PrimitiveInstr(Prims.MUL),
            mutableListOf(a, b)
        )
    }

    val outSha = listOf(shapeMul)
        .wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.RESHAPE_VIEW),
        mutableListOf(outSha, args[0])
    ))
}.parallelWithoutDeepCopy()