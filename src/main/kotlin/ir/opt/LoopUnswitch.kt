package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.log

val loopUnswitch = Pass<Unit>("loop unswitch") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEach { instr ->
        if (!a.isPrim(instr, Prims.Comp.REPEAT))
            return@forEach

        val fn = a.function(instr.args[2])
            ?: return@forEach
        val fnA = Analysis(fn)

        val switches = fn.instrs.filter { fnA.isPrim(it, Prims.SWITCH) }

        if (switches.isEmpty())
            return@forEach

        log("almost trailing switch in blk: ${fn.name}")

        val trailing = switches.firstOrNull(fnA::isLast)
            ?: return@forEach

        if (trailing.args[2] in fn.args) { // independent from repeat block = in repeat block args
            log("unswitch in ${block.name}")
            // TODO
        }
    }
}