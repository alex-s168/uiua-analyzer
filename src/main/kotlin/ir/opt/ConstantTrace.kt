package me.alex_s168.uiua.ir.opt

import blitz.collections.hasLeast
import blitz.isA
import me.alex_s168.uiua.CallerInstrsCache
import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.Type
import me.alex_s168.uiua.ir.*

private fun constTrace(a: Analysis, origlessVal: IrVar, oldInst: IrInstr, varType: Type, argIdx: Int, cache: CallerInstrsCache): IrInstr? {
    return a.deepOriginV2(origlessVal, cache::get)?.let { origE ->
        if (origE.isA()) {
            val (_, orig) = origE.assertA()
            if (orig.instr is PushFnRefInstr) {
                a.transform(listOf(oldInst)) { put, newVar ->
                    val newv = newVar().copy(type = varType)

                    put(IrInstr(
                        mutableListOf(newv),
                        PushFnRefInstr(orig.instr.fn),
                        mutableListOf()
                    ))

                    deepCopy().also { it.args[argIdx] = newv }.also(put)
                }
            } else null
        } else {
            val const = origE.assertB()

            a.transform(listOf(oldInst)) { put, newVar ->
                val newv = newVar().copy(type = varType)

                put(IrInstr(
                    mutableListOf(newv),
                    NumImmInstr(const),
                    mutableListOf()
                ))

                deepCopy().also { it.args[argIdx] = newv }.also(put)
            }
        }
    }?.first()
}

val constantTrace = Pass<Unit>("const trace") { block, _ ->
    val a = Analysis(block)
    val cache = CallerInstrsCache(4)

    if (cache.get(block).hasLeast(2))
        return@Pass

    block.instrs.toList().forEach { instr ->
        var instr = instr
        instr.args
            .filter { it in block.args }
            .forEachIndexed { argIdx, arg ->
                constTrace(a, arg, instr, arg.type, argIdx, cache)?.let {
                    instr = it
                }
            }
    }
} // TODO: is faster without parallel?