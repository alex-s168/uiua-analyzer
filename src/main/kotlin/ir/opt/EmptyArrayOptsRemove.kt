package me.alex_s168.uiua.ir.opt

import blitz.Either
import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.ir.*

private fun emptyArray(block: IrBlock, arr: IrVar, putBlock: (IrBlock) -> Unit) {
    val a = Analysis(block)

    a.usages(arr).forEach {
        it ?: return@forEach

        if (a.isPrim(it, Prim.Comp.ARR_COPY)) {
            block.instrs.remove(it)
        }
        else if (a.isPrim(it, Prim.Comp.ARR_LOAD)) {
            val idx = block.instrs.indexOf(it)
            block.instrs.add(idx, IrInstr(
                mutableListOf(),
                PrimitiveInstr(Prim.Comp.PANIC),
                mutableListOf()
            ))
        }
        else if (a.isPrim(it, Prim.Comp.ARR_STORE)) {
            if (it.args[0] == arr) {
                val idx = block.instrs.indexOf(it)
                block.instrs.add(idx, IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prim.Comp.PANIC),
                    mutableListOf()
                ))
            }
            else if (it.args[2] == arr) {
                block.instrs.remove(it)
            }
        }
        else if (a.isPrim(it, Prim.Comp.DIM)) {
            val idx = block.instrs.indexOf(it)
            block.instrs[idx] = IrInstr(
                mutableListOf(it.outs[0]),
                NumImmInstr(0.0),
                mutableListOf()
            )
        }
        else if (a.isPrim(it, Prim.Comp.REPEAT)) {
            val fnref = block.instrDeclFor(it.args[2])
            block.funDeclFor(it.args[2])?.let { (_, fn) ->
                val argIdx = it.args.indexOf(arr) - 2

                val fn = if (Analysis(fn).callerInstrs().size > 1) {
                    fn.deepCopy()
                        .also(putBlock)
                        .also {
                            (fnref!!.instr as PushFnRefInstr).fn = it.name
                        }
                } else fn

                val arg = fn.args[argIdx]
                emptyArray(fn, arg, putBlock)
            }
        }
    }
}

val emptyArrayOpsRemove = Pass<(IrBlock) -> Unit>("infer array sizes") { block, putBlock ->
    val a = Analysis(block)

    block.instrs.toList().forEach {
        if (a.isPrim(it, Prim.Comp.ARR_ALLOC)) {
            val shape = block.instrDeclFor(it.args[0])
                ?.args
                ?: return@forEach

            val shapec = shape.map {
                a.deepOrigin(it)
                    ?.let { (_, i) -> if (i.instr is NumImmInstr) Either.ofA(i.instr.value) else null }
                    ?: Either.ofB(it)
            }

            val arr = it.outs[0]

            if (shapec.none { it.isA && it.getA() == 0.0 })
                return@forEach

            emptyArray(block, arr, putBlock)
        }
    }
}