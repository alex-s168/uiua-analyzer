package me.alex_s168.uiua.ir.opt

import blitz.mapA
import blitz.mapB
import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.CallerInstrsCache
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.comment
import me.alex_s168.uiua.ir.transform.passValue

// TODO: replace indexOf with idx in LightCache

val evalDim = Pass<Unit>("evaluate dim") { block, _ ->
    val cache = CallerInstrsCache(0)

    val a = Analysis(block)

    repeat(block.instrs.size) { index ->
        val instr = block.instrs[index]
        if (a.isPrim(instr, Prims.Comp.DIM)) {
            val arr = instr.args[0]

            val dim = a.constNum(instr.args[1], cache::get)
                ?.toInt()
                ?: return@repeat

            val shapec = a.constShape(arr, cache::get)
                ?: return@repeat

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
                Unit
            }
        }
    }
}