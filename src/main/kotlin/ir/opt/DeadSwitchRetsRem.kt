package me.alex_s168.uiua.ir.opt

import blitz.collections.gather
import blitz.collections.hasLeast
import blitz.collections.removeAtIndexes
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.debugVerify
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val deadRetsRem = Pass<Unit>("dead rets rem") { block, _ ->
    val a = Analysis(block)

    block.instrs.toList().forEach { inst ->
        if (!a.isPrim(inst, Prims.SWITCH))
            return@forEach

        val dests = a.origin(inst.args[1])!!.args
            .map { a.function(it) ?: return@forEach }

        if (debugVerify) {
            require(dests.none { Analysis(it).callerInstrs().hasLeast(2) }) {
                "you need to run oneblockonecaller before deadretsrem"
            }
        }

        val toRem = inst.outs.mapIndexedNotNull { index, it ->
            if (a.usages(it).all { it != null && a.isPrim(it, Prims.Comp.SINK) }) {
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
        }
    }
}.parallelWithoutDeepCopy()