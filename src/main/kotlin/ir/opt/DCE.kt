package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.dontCareOpsBeforePanic
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Pass

val dce = Pass<Unit>("DCE") { block, _ ->
    val a = Analysis(block)

    if (dontCareOpsBeforePanic) {
        if (a.terminating()) {
            block.instrs.clear()
            block.instrs.add(IrInstr(
                block.rets.toMutableList(),
                PrimitiveInstr(Prim.Comp.PANIC),
                mutableListOf()
            ))
        }
    } else {
        val idx = block.instrs.indexOfFirst { a.terminating(it) }
        if (idx < 0) return@Pass

        repeat(block.instrs.size - idx) {
            block.instrs.removeAt(idx)
        }

        val newOuts = block.rets.mapTo(mutableListOf()) { block.newVar().copy(type = it.type) }
        block.rets = newOuts

        block.instrs.add(IrInstr(
            block.rets.toMutableList(),
            PrimitiveInstr(Prim.Comp.PANIC),
            mutableListOf()
        ))
    }
}