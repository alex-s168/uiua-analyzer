package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Pass

val dce = Pass<Unit>("DCE") { block, _ ->
    val a = Analysis(block)

    val idx = block.instrs.indexOfFirst { a.isPrim(it, Prim.Comp.PANIC) }
    if (idx < 0) return@Pass

    repeat(block.instrs.size - idx) {
        block.instrs.removeAt(idx)
    }

    val newOuts = block.rets.mapTo(mutableListOf()) { block.newVar().copy(type = it.type) }
    block.rets = newOuts

    block.rets.forEach {
        block.instrs.add(IrInstr(
            mutableListOf(it),
            PrimitiveInstr(Prim.Comp.PANIC),
            mutableListOf()
        ))
    }
}