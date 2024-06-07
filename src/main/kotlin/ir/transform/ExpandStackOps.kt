package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.expandStackOps() {
    var i = 0
    while (i < instrs.size) {
        val instr = instrs[i]
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.DUP -> {
                    val inp = instr.args[0]

                    instrs.removeAt(i)
                    i--
                    instr.outs.forEach {
                        instrs.add(i, IrInstr(
                            mutableListOf(it),
                            PrimitiveInstr("cUSE"),
                            mutableListOf(inp),
                        ))
                        i++
                    }
                }
                Prim.FLIP -> {
                    instrs.removeAt(i)
                    instrs.add(i, IrInstr(
                        mutableListOf(instr.outs[0]),
                        PrimitiveInstr("cUSE"),
                        mutableListOf(instr.args[1]),
                    ))
                    instrs.add(i + 1, IrInstr(
                        mutableListOf(instr.outs[1]),
                        PrimitiveInstr("cUSE"),
                        mutableListOf(instr.args[0]),
                    ))
                    i++
                }
            }
        }
        i++
    }
}