package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

private fun lowerEachRec(
    newVar: () -> IrVar,
    fill: IrVar?,
    ref: (String) -> IrBlock?,
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
            PrimitiveInstr(Prim.CALL),
            mutableListOf(innerFn).also { it += inputs }
        ))
        return
    }

    val block = IrBlock(anonFnName(), ref).apply {
        fillArg = fill?.let { newVar().copy(type = it.type) }

        val innerFn = newVar().copy(type = innerFn.type).also { args += it }
        val inputs = inputs.map {
            val ty = if (it.type !is ArrayType || it.type.length == 1) it.type
            else it.type.of
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
        PushFnRefInstr(block.name),
        mutableListOf()
    ))

    put(IrInstr(
        outputs.toMutableList(),
        PrimitiveInstr(Prim.ROWS),
        mutableListOf(blockFn, innerFn).also { it += inputs }
    ))
}

fun IrBlock.lowerEach(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.EACH -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    instrs.add(idx ++, comment("+++ each"))

                    val fn = instr.args[0]

                    val highestRank = instr.args.drop(1)
                        .filter { it.type is ArrayType && it.type.length != 1 }
                        .maxBy { (it.type as ArrayType).shape.size }
                        .type as ArrayType

                    lowerEachRec(::newVar, fillArg, ref, {
                        instrs.add(idx ++, it)
                    }, putBlock, instr.args.drop(1), instr.outs, fn, highestRank.shape.size)

                    instrs.add(idx ++, comment("--- each"))
                }
            }
        }
    }
}