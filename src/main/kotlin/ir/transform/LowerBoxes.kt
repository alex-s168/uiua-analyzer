package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

private fun Type.convBox(): Type =
    when (this) {
        is BoxType -> Types.array(of, 1).also { it.convBox() }
        is PtrType -> Types.pointer(to.convBox())
        is ArrayType -> Types.array(of, length)
        else -> this
    }

private fun MutableList<IrVar>.convBox() {
    forEachIndexed { index, o ->
        this[index] = o.copy(type = o.type.convBox())
    }
}

fun IrBlock.lowerBoxesToArrays() {
    args.convBox()
    rets.convBox()

    instrs.toList().forEach { instr ->
        instr.outs.convBox()
        instr.args.convBox()

        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.Comp.BOX_CREATE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val shape = constantArr(::newVar, 1.0, type = Types.int) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(instr.outs[0]),
                        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                        mutableListOf(shape)
                    ))
                }

                Prim.Comp.BOX_DESTROY -> {
                    val idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)
                    instrs.add(idx, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.ARR_DESTROY),
                        mutableListOf(instr.args[0])
                    ))
                }

                Prim.Comp.BOX_LOAD -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    oneDimLoad(dest = instr.outs[0], instr.args[0], ::newVar, 0) {
                        instrs.add(idx ++, it)
                    }
                }

                Prim.Comp.BOX_STORE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val indecies = constantArr(::newVar, 0.0, type = Types.int) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.ARR_STORE),
                        mutableListOf(instr.args[0], indecies, instr.args[1])
                    ))
                }
            }
        }
    }
}