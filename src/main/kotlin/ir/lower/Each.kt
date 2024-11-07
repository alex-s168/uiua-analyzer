package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.ir.lowerPrimPass

private fun lowerEachRec(
    newVar: () -> IrVar,
    fill: IrVar?,
    ref: Map<BlockId, IrBlock>,
    put: (IrInstr) -> Unit,
    putBlock: (IrBlock) -> Unit,
    inputs: List<IrVar>,
    outputs: List<IrVar>,
    innerFn: IrVar,
    left: Int
) {
    if (left == 0) {
        put(IrInstr(
            outputs.toMutableList(),
            PrimitiveInstr(Prims.CALL),
            mutableListOf(innerFn).also { it += inputs }
        ))
        return
    }

    val block = IrBlock(anonFnName(), ref).apply {
        fillArg = fill?.let { newVar().copy(type = it.type) }

        val innerFn = newVar().copy(type = innerFn.type).also { args += it }
        val inputs = inputs.map {
            val ty = if (it.type !is ArrayType || it.type.length == 1) it.type
            else it.type.of.makeVaOffIfArray()
            newVar().copy(type = ty).also { args += it }
        }

        val outputs = outputs.map {
            val ty = if (it.type is ArrayType) it.type.of
            else it.type
            newVar().copy(type = ty).also { rets += it }
        }

        lowerEachRec(::newVar, fillArg, ref, { instrs += it }, putBlock, inputs, outputs, innerFn, left - 1)

        putBlock(this)
    }

    val blockFn = newVar().copy(type = block.type())
    put(IrInstr(
        mutableListOf(blockFn),
        PushFnRefInstr(block.uid),
        mutableListOf()
    ))

    put(IrInstr(
        outputs.toMutableList(),
        PrimitiveInstr(Prims.ROWS),
        mutableListOf(blockFn, innerFn).also { it += inputs }
    ))
}

val lowerEach = lowerPrimPass<(IrBlock) -> Unit>(Prims.EACH) { put, newVar, a, putBlock ->
    val fn = args[0]

    val highestRank = args.drop(1)
        .filter { it.type is ArrayType && it.type.length != 1 }
        .maxBy { (it.type as ArrayType).shape.size }
        .type as ArrayType

    lowerEachRec(newVar, a.block.fillArg, a.block.ref, put, putBlock, args.drop(1), outs, fn, highestRank.shape.size)
}