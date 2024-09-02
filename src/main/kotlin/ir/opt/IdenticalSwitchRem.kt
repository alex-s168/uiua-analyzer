package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass

val identicalSwitchRem = Pass<Unit>("identical switch rem") { block, _ ->
    val a = Analysis(block)

    block.instrs.toList().forEach { inst ->
        if (a.isPrim(inst, Prim.SWITCH)) {
            val dests = a.origin(inst.args[1])!!.args
                .map { a.function(it) ?: return@forEach }

            if (dests.isEmpty())
                return@forEach

            if (!dests.drop(1).all { Analysis.blocksEqual(it, dests[0]) })
                return@forEach

            a.transform(listOf(inst)) { put, newVar ->
                val inl = dests[0].inlinableCopy(inst.args.drop(3), inst.outs, block.fillArg)
                inl.instrs.forEach(put)
            }
        }
    }

    a.finish("identical switch rem")
}