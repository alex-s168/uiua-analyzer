package me.alex_s168.uiua.ir.opt

import blitz.collections.gather
import blitz.collections.removeAtIndexes
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.debugVerify
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val deadRetsRem = Pass<Unit>("dead rets rem") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEach { inst ->
        if (!a.isPrim(inst, Prim.SWITCH))
            return@forEach

        val dests = a.origin(inst.args[1])!!.args
            .map { a.function(it) ?: return@forEach }

        if (debugVerify) {
            require(dests.all { Analysis(it).callerInstrs().size == 1 }) {
                "you need to run oneblockonecaller before deadretsrem"
            }
        }

        val toRem = inst.outs.mapIndexedNotNull { index, it ->
            if (a.usages(it).all { it != null && a.isPrim(it, Prim.Comp.SINK) }) {
                index
            } else null
        }

        inst.outs.gather(toRem).flatMap(a::usages).forEach {
            block.instrs -= it!!
        }

        inst.outs.removeAtIndexes(toRem).let { r ->
            inst.outs.clear()
            inst.outs += r
        }

        dests.forEach {
            it.rets = it.rets.removeAtIndexes(toRem)
            Analysis(it).updateFnType()
        }
    }
}.parallelWithoutDeepCopy()