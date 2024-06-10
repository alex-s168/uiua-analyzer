package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerSimple() {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.LEN -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    instrs.add(idx ++, comment("+++ len"))

                    if (instr.args[0].type is ArrayType) {
                        val const = newVar().copy(type = Types.size)
                        instrs.add(idx++, IrInstr(
                            mutableListOf(const),
                            NumImmInstr(0.0),
                            mutableListOf()
                        ))

                        instrs.add(idx++, IrInstr(
                            instr.outs,
                            PrimitiveInstr(Prim.Comp.DIM),
                            mutableListOf(instr.args[0], const)
                        ))
                    }
                    else {
                        instrs.add(idx++, IrInstr(
                            mutableListOf(instr.outs[0]),
                            NumImmInstr(1.0),
                            mutableListOf()
                        ))
                    }

                    instrs.add(idx ++, comment("--- len"))
                }
            }
        }
    }
}