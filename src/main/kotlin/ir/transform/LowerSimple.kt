package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerSimple() {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.LEN -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val const = newVar().copy(type = Types.int)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(const),
                        NumImmInstr(0.0),
                        mutableListOf()
                    ))

                    instrs.add(idx ++, IrInstr(
                        instr.outs,
                        PrimitiveInstr(Prim.Comp.DIM),
                        mutableListOf(instr.args[0], const)
                    ))
                }
            }
        }
    }
}