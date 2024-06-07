package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerBoxesToArrays() {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.Comp.BOX_CREATE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val shape = newVar().copy(type = Types.array(Types.int))
                    constantArr(shape, ::newVar, 1.0, type = Types.int) {
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

                    val indecies = newVar().copy(type = Types.array(Types.int))
                    constantArr(indecies, ::newVar, 0.0, type = Types.int) {
                        instrs.add(idx ++, it)
                    }

                    instrs.add(idx ++, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prim.Comp.ARR_LOAD),
                        mutableListOf(instr.args[0], indecies)
                    ))
                }

                Prim.Comp.BOX_STORE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val indecies = newVar().copy(type = Types.array(Types.int))
                    constantArr(indecies, ::newVar, 0.0, type = Types.int) {
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