package me.alex_s168.uiua.ir.opt

import blitz.unreachable
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.withoutParallel

val argRem = Pass<Unit>("arg rem") { block, _ ->
    val a = Analysis(block)

    val unusedArgs = block.args
        .filter(a::unused)
        .map { block.args.indexOf(it) }

    if (unusedArgs.isEmpty())
        return@Pass

    val callers = a.callerInstrs()

    val removable = BooleanArray(unusedArgs.size) { true }

    callers.forEach { (caller, instr) ->
        if (a.isPrim(instr, Prim.SWITCH)) {
            val ca = Analysis(caller)
            val calling = ca.origin(instr.args[1])!!.args
            for (c in calling) {
                val fn = ca.function(c) ?: return@Pass
                if (fn != block) {
                    require(fn.args.size == block.args.size) {
                        "cases in switch have different args"
                    }
                    val fna = Analysis(fn)
                    unusedArgs.forEachIndexed { idx, it ->
                        if (removable[idx]) {
                            if (!fna.unused(fn.args[it]))
                                removable[idx] = false
                        }
                    }
                }
            }
        }
    }

    if (removable.none())
        return@Pass

    val remua = unusedArgs
        .filterIndexed { idx, _ -> removable[idx] }

    remua
        .map { block.args[it] }
        .forEach(block.args::remove)

    callers.forEach { (caller, callerInst) ->
        require(callerInst.instr is PrimitiveInstr)
        val instrCallArgsBegin = when (callerInst.instr.id) {
            Prim.SWITCH -> 3
            Prim.Comp.REPEAT -> 2 // don't change!!!
            Prim.CALL -> 1
            else -> unreachable()
        }

        remua.map {
            callerInst.args[instrCallArgsBegin + it]
        }.forEach {
            callerInst.args.remove(it)
        }

        if (callerInst.instr.id == Prim.SWITCH) {
            val ca = Analysis(caller)
            val calling = ca.origin(callerInst.args[1])!!
                .args
                .map(ca::function)

            calling.forEach { c ->
                c!!
                if (c != block) {
                    remua
                        .map { c.args[it] }
                        .forEach(c.args::remove)
                }
            }
        }
    }
}.withoutParallel()