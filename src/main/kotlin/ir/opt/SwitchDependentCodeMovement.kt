package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass

val switchDependentCodeMovement = Pass<Unit>("Switch Dependent Code Movement") { block, _ ->
    val a = Analysis(block)

    val switchStmts = block.instrs
        .filter { v -> v.instr is PrimitiveInstr && v.instr.id == Prim.SWITCH }
        .filter { it.outs.isNotEmpty() }

    switchStmts.forEach { switch ->
        val move = switch.outs.flatMap { i -> a.dependentCodeBlockAsMovable(i, switch) }.distinct()

        if (move.isEmpty())
            return@forEach

        // we already moved it
        if (switch !in block.instrs)
            return@forEach

        if (move.any { it !in block.instrs })
            return@forEach

        val deps = a.allDependencies(move)
            .filter { it !in switch.outs }

        val dests = block.instrDeclFor(switch.args[1])
            ?.args
            ?.map { block.funDeclFor(it)!!.second }
            ?: return@forEach

        if (dests.any { Analysis(it).callers().size > 1 })
            return@forEach

        // =====================================

        val switchOldOuts = switch.outs.toList()

        val additionalOuts = move.flatMap { it.outs }.mapNotNull { v ->
            if (v in block.rets) {
                val copy = block.newVar().copy(type = v.type)
                switch.outs.add(copy)
                block.rets[block.rets.indexOf(v)] = copy
                v
            } else null
        }
        println("add outs : $additionalOuts")

        block.instrs.removeAll(move)

        switch.args.addAll(deps)

        println("moving:")
        move.forEach {
            println("  $it")
        }

        dests.forEach { dest ->
            println(dest)

            val newDeps = deps.map { dep ->
                dest.newVar().copy(type = dep.type)
                    .also(dest.args::add)
            }

            val outs = move.flatMap {
                val instr = it.deepCopy()
                    .also(dest.instrs::add)

                instr.args.replaceAll {
                    val depIdx = deps.indexOf(it)
                    if (depIdx == -1) it
                    else newDeps[depIdx]
                }

                instr.outs
            }

            val outsMap = outs.associateWith {
                val new = dest.newVar().copy(type = it.type)
                dest.updateVar(it, new)
                new
            }

            switchOldOuts.forEachIndexed { i, old ->
                dest.updateVar(old, dest.rets[i])
            }

            additionalOuts.forEach {
                dest.rets += outsMap[it]!!
            }

            Analysis(dest).updateFnType()
        }
    }
}