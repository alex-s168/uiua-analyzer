package me.alex_s168.uiua.ir.opt

import blitz.logic.then
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrBlock

fun IrBlock.optInlineCUse() {
    instrs.removeIf {
        (it.instr is PrimitiveInstr && it.instr.id == Prim.Comp.USE).then {
            updateVar(it.outs[0], it.args[0])
        }
    }
}