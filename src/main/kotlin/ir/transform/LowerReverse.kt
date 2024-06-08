package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.BoxType
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerReverse(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.REVERSE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val arg = instr.args[0]

                    val operate = if (arg.type is BoxType) {
                        val unboxed = newVar().copy(type = arg.type.of)
                        instrs.add(idx ++, IrInstr(
                            mutableListOf(unboxed),
                            PrimitiveInstr(Prim.UN_BOX),
                            mutableListOf(arg)
                        ))
                        unboxed
                    } else arg

                    val out = if (arg.type is BoxType) {
                        newVar().copy(type = arg.type.of)
                    } else instr.outs[0]

                    val len = newVar().copy(type = Types.size)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(len),
                        PrimitiveInstr(Prim.LEN),
                        mutableListOf(operate),
                    ))

                    val shape = listOf(len).wrapInArgArray(::newVar) { instrs.add(idx ++, it) }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(out),
                        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                        mutableListOf(shape)
                    ))

                    // TODO: reverse into out

                    out.into(instr.outs[0]) { instrs.add(idx ++, it) }
                }
            }
        }
    }
}