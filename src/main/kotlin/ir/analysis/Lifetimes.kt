package me.alex_s168.uiua.ir.analysis

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.*

private data class ArgLifetimeAnalysis(
    val visited: MutableList<IrBlock>
)

private fun IrBlock.updateArgLifetimes(ctx: ArgLifetimeAnalysis) {
    val a = Analysis(this)

    ctx.visited += this
    a.variables().forEach { arg ->
        lifetimes[arg] = Lifetime.LOCAL
        a.usages(arg).filterNotNull().forEach { usage ->
            if (usage.instr is PrimitiveInstr) {
                when (usage.instr.id) {
                    Prims.CALL -> {
                        lifetimes[arg] = a.function(usage.args[0])?.let {
                            if (it !in ctx.visited) it.updateArgLifetimes(ctx)
                            it.lifetimes[arg]
                        } ?: Lifetime.GLOBAL
                    }

                    Prims.Comp.BOX_STORE -> {
                        if (arg == usage.args[1]) {
                            lifetimes[arg] = Lifetime.GLOBAL
                        }
                    }

                    Prims.Comp.ARR_STORE -> {
                        if (arg == usage.args[2]) {
                            lifetimes[arg] = Lifetime.GLOBAL
                        }
                    }
                }
            }
        }
    }
}

val lifetimes = analysisPass("lifetimes") { block, a ->
    block.updateArgLifetimes(ArgLifetimeAnalysis(mutableListOf()))
}.parallelWithoutDeepCopy()