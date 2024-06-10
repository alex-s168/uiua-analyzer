package me.alex_s168.uiua.ir.transform

import blitz.flatten
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.expandArrays() {
    instrs.toList().forEach { instr ->
        if (instr.instr is ArrImmInstr) {
            var idx = instrs.indexOf(instr)
            instrs.removeAt(idx)

            instrs.add(idx ++, comment("+++ array literal"))

            val values = instr.instr.values
                .mapA { it.map { it.toDouble() } }
                .flatten()

            val arrType = instr.outs[0].type as ArrayType

            val arr = newVar().copy(type = arrType)
            constantArr(arr, ::newVar, *values.toDoubleArray()) {
                instrs.add(idx ++, it)
            }

            instrs.add(idx ++, IrInstr(
                mutableListOf(instr.outs[0]),
                PrimitiveInstr(Prim.Comp.ARR_MATERIALIZE),
                mutableListOf(arr)
            ))

            instrs.add(idx ++, comment("--- array literal"))
        }
    }
}