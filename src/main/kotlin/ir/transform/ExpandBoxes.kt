package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.expandBoxes() {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.BOX -> {
                    val idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)
                    instrs.add(idx, IrInstr(
                        instr.outs,
                        PrimitiveInstr(Prim.Comp.BOX_CREATE),
                        mutableListOf()
                    ))
                    instrs.add(idx + 1, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.BOX_STORE),
                        mutableListOf(instr.outs[0], instr.args[0])
                    ))
                }

                Prim.UN_BOX -> {
                    val idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)
                    instrs.add(idx, IrInstr(
                        instr.outs,
                        PrimitiveInstr(Prim.Comp.BOX_LOAD),
                        instr.args
                    ))
                }
            }
        }
    }
}