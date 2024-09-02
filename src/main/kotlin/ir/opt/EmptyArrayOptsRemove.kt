package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.*

private fun emptyArray(block: IrBlock, arr: IrVar, putBlock: (IrBlock) -> Unit) {
    val a = Analysis(block)

    a.usages(arr).forEach {
        it ?: return@forEach // TODO: emptyArray down (inside->out)

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
        else {
            a.getDeepCalling(it).forEach { fnRef ->
                var argIdx = it.args.indexOf(arr)
                if (argIdx == -1) return@forEach // TODO: wtf
                argIdx -= when ((it.instr as PrimitiveInstr).id) {
                    Prim.Comp.REPEAT -> 2
                    Prim.CALL -> 1
                    Prim.SWITCH -> 3
                    else -> return@forEach
                }

                a.function(fnRef.get())?.let { fn ->
                    val fnA = Analysis(fn)
                    if (fnA.callerInstrs().size > 1)
                        error("you need to run oneblockonecaller before emptyArrayOpsRemove")

                    val arg = fn.args[argIdx]
                    println("tracing empty array from ${block.name} -> ${fn.name}")
                    emptyArray(fn, arg, putBlock)
                }
            }
        }
    }
}

val emptyArrayOpsRemove = Pass<(IrBlock) -> Unit>("infer array sizes") { block, putBlock ->
    val a = Analysis(block)

    block.instrs.toList().forEach {
        if (a.isPrim(it, Prim.Comp.ARR_ALLOC)) {
            val arr = it.outs[0]
            val shapec = a.constShape(arr)
                ?: return@forEach

            if (shapec.none { it.isA && it.getA() == 0 })
                return@forEach

            println("empty array: $arr")
            emptyArray(block, arr, putBlock)
        }
    }
}