package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerArrMat() {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.Comp.ARR_MATERIALIZE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    instrs.add(idx ++, comment("+++ array materialize"))

                    val data = instrDeclFor(instr.args[0])!!.args

                    val shape = constantArr(::newVar, data.size.toDouble(), type = Types.size) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(instr.outs[0]),
                        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                        mutableListOf(shape)
                    ))

                    data.forEachIndexed { index, src ->
                        val idc = constantArr(::newVar, index.toDouble(), type = Types.size) {
                            instrs.add(idx ++, it)
                        }
                        instrs.add(idx ++, IrInstr(
                            mutableListOf(),
                            PrimitiveInstr(Prim.Comp.ARR_STORE),
                            mutableListOf(instr.outs[0], idc, src)
                        ))
                    }

                    instrs.add(idx ++, comment("--- array materialize"))
                }
            }
        }
    }
}