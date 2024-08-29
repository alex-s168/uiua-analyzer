package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.fullUnrollLoop
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.transform.constants

val unrollLoop = Pass<Unit>("full unroll loop") { block, _ ->
    val a = Analysis(block)

    block.instrs.toList().forEach { instr ->
        if (!a.isPrim(instr, Prim.Comp.REPEAT))
            return@forEach

        val begin = a.constNum(instr.args[0])
            ?.toInt()
            ?: return@forEach

        val end = a.constNum(instr.args[1])
            ?.toInt()
            ?: return@forEach

        val count = end - begin + 1

        if (!fullUnrollLoop(block, count))
            return@forEach

        val fn = a.function(instr.args[2])
            ?: return@forEach

        val extra = instr.args.drop(3)

        a.transform(listOf(instr)) { put, newVar ->
            var iter = begin
            while (iter <= end) {
                val (iterConst) = constants(newVar, iter.toDouble(), type = Types.size, put = put)

                val inline = fn.inlinableCopy(
                    cArgs = listOf(iterConst) + extra,
                    cRets = listOf(),
                    fill = block.fillArg,
                )

                inline.instrs.forEach(put)

                iter ++
            }
        }
    }

    a.finish("full unroll loop")
}