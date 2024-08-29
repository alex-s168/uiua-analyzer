package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.transform.comment
import me.alex_s168.uiua.ir.transform.passValue

val evalDim = Pass<Unit>("evaluate dim") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEachIndexed { index, instr ->
        if (a.isPrim(instr, Prim.Comp.DIM)) {
            val arr = instr.args[0]

            val dim = a.constNum(instr.args[1])
                ?.toInt()
                ?: return@forEachIndexed

            val shapec = a.constShape(arr)
                ?: return@forEachIndexed

            val res = shapec[dim]
            res.mapA {
                block.instrs[index] = IrInstr(
                    instr.outs,
                    NumImmInstr(it.toDouble()),
                    mutableListOf()
                )
            }.mapB { bv ->
                passValue(bv, block)?.let {
                    a.rename(instr.outs[0], it)
                    block.instrs[index] = comment("rem")
                }
            }
        }
    }
}