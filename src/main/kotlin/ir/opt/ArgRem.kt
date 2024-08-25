package me.alex_s168.uiua.ir.opt

import blitz.collections.contents
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass

val argRem = Pass<Unit>("arg rem") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEach { instr ->
        if (instr.instr !is PrimitiveInstr) return@forEach

        val instrOff = when (instr.instr.id) {
            Prim.SWITCH -> 3
            Prim.CALL -> 1
            Prim.Comp.REPEAT -> 2 // +3 -1  (-1 bc takes counter)
            else -> return@forEach
        }

        val dests = when (instr.instr.id) {
            Prim.SWITCH -> (a.origin(instr.args[1]) ?: return@forEach).args
                .map { block.funDeclFor(it)?.second }
            Prim.CALL -> listOf((block.funDeclFor(instr.args[0]) ?: return@forEach).second)
            Prim.Comp.REPEAT -> listOf((block.funDeclFor(instr.args[2]) ?: return@forEach).second)
            else -> return@forEach
        }

        if (dests.isNotEmpty() && dests.none { it == null } && dests.all { Analysis(it!!).callerInstrs().contents == arrayOf(block to instr).contents }) {
            val removable = dests.map { dest ->
                dest!!

                dest.args
                    .filter { !dest.varUsed(it) }
                    .map(dest.args::indexOf)
            }

            val remove = removable
                .reduce { acc, ints -> acc.intersect(ints).toList() }

            if (remove.isNotEmpty()) {
                dests.forEach {
                    it!!.args = it.args.filterIndexedTo(mutableListOf()) { index, _ -> index !in remove }
                }

                instr.args = instr.args.filterIndexedTo(mutableListOf()) { index, _ -> (index + instrOff) !in remove }

                println("removed some args")
            }
        }
    }
}