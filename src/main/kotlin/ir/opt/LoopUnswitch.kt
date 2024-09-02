package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass

val loopUnswitch = Pass<Unit>("loop unswitch") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEach { instr ->
        if (!a.isPrim(instr, Prim.Comp.REPEAT))
            return@forEach

        val fn = a.function(instr.args[2])
            ?: return@forEach
        val fnA = Analysis(fn)

        val switches = fn.instrs.filter { fnA.isPrim(it, Prim.SWITCH) }

        if (switches.isEmpty())
            return@forEach

        println("almost trailing switch in blk: ${fn.name}")

        val trailing = switches.firstOrNull(fnA::isLast)
            ?: return@forEach

        if (trailing.args[2] in fn.args) { // independent from repeat block = in repeat block args
            println("unswitch in ${block.name}")
            // TODO
        }
    }
}