package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.Type
import me.alex_s168.uiua.ir.*

private fun constTrace(a: Analysis, origlessVal: IrVar, oldInst: IrInstr, varType: Type, argIdx: Int): IrInstr? {
    return a.deepOriginV2(origlessVal)?.let { origE ->
        if (origE.isA) {
            val (_, orig) = origE.getA()
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
            val const = origE.getB()

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

    require(a.callerInstrs().size <= 1) {
        "you forgot to run oneBlockOneCaller pass before constantTrace pass"
    }

    block.instrs.toList().forEach { instr ->
        var instr = instr
        instr.args.forEachIndexed { argIdx, arg ->
            if (arg in block.args) {
                println("constrace in block ${block.name} instr: $instr")
                constTrace(a, arg, instr, arg.type, argIdx)?.let {
                    instr = it
                }
            }
        }
    }
}