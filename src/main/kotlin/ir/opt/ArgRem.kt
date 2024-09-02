package me.alex_s168.uiua.ir.opt

import blitz.collections.contents
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.intersections
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Pass

val argRem = Pass<Unit>("arg rem") { block, _ ->
    val a = Analysis(block)

    block.instrs.toList().forEach { instr ->
        if (instr.instr !is PrimitiveInstr)
            return@forEach

        val instrCallArgsBegin = when (instr.instr.id) {
            Prim.SWITCH -> 3
            Prim.Comp.REPEAT -> 2 // don't change!!!
            Prim.CALL -> 1
            else -> return@forEach
        }

        val dests = when (instr.instr.id) {
            Prim.SWITCH -> (a.origin(instr.args[1]) ?: return@forEach).args
                .map { block.funDeclFor(it)?.second }
            Prim.CALL -> listOf(block.funDeclFor(instr.args[0])?.second)
            Prim.Comp.REPEAT -> listOf(block.funDeclFor(instr.args[2])?.second)
            else -> return@forEach
        }

        if (dests.isEmpty()) return@forEach
        if (dests.any { it == null } ) return@forEach
        if (!dests.all { Analysis(it!!).callerInstrs().contents == arrayOf(block to instr).contents }) return@forEach

        val removable = dests.map { dest ->
            dest!!

            dest.args
                .filter { !dest.varUsed(it) }
                .map(dest.args::indexOf)
        }

        val remove = removable.intersections()
        if (a.isPrim(instr, Prim.Comp.REPEAT)) {
            remove -= 0 // can't remove counter
        }
        if (remove.isEmpty()) return@forEach

        a.transform(listOf(instr)) { put, newVar ->
            val undef = remove.map { newVar().copy(type = args[instrCallArgsBegin + it].type) }

            put(IrInstr(
                undef.toMutableList(),
                PrimitiveInstr(Prim.Comp.UNDEF),
                mutableListOf()
            ))

            put(this.deepCopy().also { inst ->
                remove.zip(undef).forEach { (it, new) ->
                    inst.args[instrCallArgsBegin + it] = new
                }
            })
        }
    }
}