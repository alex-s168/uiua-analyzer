package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass

val loopUnswitch = Pass<Unit>("loop unswitch") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEach { instr ->
        if (!a.isPrim(instr, Prim.Comp.REPEAT))
            return@forEach

        val fn = a.function(instr.args[0])
            ?: return@forEach

        val switches = fn.instrs.filter { a.isPrim(it, Prim.SWITCH) }

        println("almosst almost")

        if (switches.isEmpty() || switches.size > 1)
            return@forEach

        println("almost trailing switch")

        if (!Analysis(fn).isLast(switches[0]))
            return@forEach

        println("trailing switch in ${block.name}")
    }
}