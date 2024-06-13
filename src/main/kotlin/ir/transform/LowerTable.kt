package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrBlock

fun IrBlock.lowerTable(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.TABLE -> {
                    // TODO: lower to    rows(rows F fix) A fix B    (only works for two args table)

                }
            }
        }
    }
}