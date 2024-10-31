package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.log

// TODO: finish

// loop independent code movement
val licm = Pass<Unit>("licm") { block, _ ->
    val a = Analysis(block)

    block.instrs.forEach { instr ->
        if (!a.isPrim(instr, Prims.Comp.REPEAT))
            return@forEach

        val fn = a.function(instr.args[2])
            ?: return@forEach
        val fnA = Analysis(fn)

        val deps = fn.instrs.map { f ->
            fnA.recDependencies(listOf(f))
                .let { it.first to (it.second + f).filterNot(fnA::isConstant) }
        }

        val ok = mutableListOf<Int>()
        deps.forEachIndexed { index, (vars, insts) ->
            if (fn.args[0] !in vars) { // can't depend on index
                if (insts.isNotEmpty() && insts.all(fnA::independentOfArrayData) && fnA.canMove(insts.toList())) {
                    ok += index
                }
            }
        }

        ok.forEach { okIdx ->
            val (depVarsOld, depInsts) = deps[okIdx]
            val depVars = depVarsOld - depInsts.flatMapTo(mutableSetOf()) { it.outs }
            log("$depInsts")
        }

        log("licmable: $ok")
    }
}