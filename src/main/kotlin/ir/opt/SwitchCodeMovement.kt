package me.alex_s168.uiua.ir.opt

import blitz.collections.contents
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Pass

fun switchMove(a: Analysis, block: IrBlock, switch: IrInstr, move: List<IrInstr>, putBlock: (IrBlock) -> Unit) {
    if (move.isEmpty())
        return

    // we already moved it
    if (switch !in block.instrs)
        return

    if (move.any { it !in block.instrs })
        return

    val deps = a.allDependencies(move)
        .filter { it !in switch.outs }

    val destRefList = block.instrDeclFor(switch.args[1])

    var dests = destRefList
        ?.args
        ?.map { block.funDeclFor(it)!!.second }
        ?: return

    dests = dests.map { it.deepCopy().also(putBlock) }

    destRefList.args = dests.mapIndexedTo(mutableListOf()) { i, it ->
        val v = block.newVar().copy(type = destRefList.args[i].type)
        block.instrs.add(block.instrs.indexOf(switch), IrInstr(
            mutableListOf(v),
            PushFnRefInstr(it.name),
            mutableListOf()
        ))
        v
    }

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

    block.instrs.removeAll(move)

    switch.args.addAll(deps)

    println("moving (deps: ${deps.contents}):")
    move.forEach {
        println("  $it")
    }

    dests.forEach { dest ->
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

    val swidx = block.instrs.indexOf(switch)
    switchOldOuts.filter { !block.varUsed(it) }.forEach {
        block.instrs.add(swidx + 1, IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prim.Comp.SINK),
            mutableListOf(it)
        ))
    }
}

val switchDependentCodeMovement = Pass<(IrBlock) -> Unit>("Switch Dependent Code Movement") { block, putBlock ->
    val a = Analysis(block)

    val switchStmts = block.instrs
        .filter { a.isPrim(it, Prim.SWITCH) }
        .filter { it.outs.isNotEmpty() }

    switchStmts.forEach { switch ->
        val move = switch.outs
            .flatMap { i -> a.dependentCodeBlockAsMovable(i, switch) }
            .distinct()

        switchMove(a, block, switch, move, putBlock)
    }
}

val switchIndependentTrailingCodeMovement = Pass<(IrBlock) -> Unit>("Switch Independent Trailing Code Movement") { block, putBlock ->
    val a = Analysis(block)

    val switch = block.instrs
        .lastOrNull { a.isPrim(it, Prim.SWITCH) }
        ?: return@Pass

    val switchIdx = block.instrs.indexOf(switch)
    val move = block.instrs
        .filterIndexed { index, _ -> index > switchIdx }
        .filter { !a.isPrim(it, Prim.Comp.SINK) }

    switchMove(a, block, switch, move, putBlock)
}