package me.alex_s168.uiua.ir.opt

import blitz.collections.hasLeast
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.*

private fun emptyArray(block: IrBlock, arr: IrVar, putBlock: (IrBlock) -> Unit) {
    val a = Analysis(block)

    a.usages(arr)
        .filterNotNull()
        .toList()
        .forEach {
            if (a.isPrim(it, Prims.Comp.ARR_COPY)) {
                block.instrs.remove(it)
            }
            else if (a.isPrim(it, Prims.Comp.ARR_LOAD)) {
                val idx = block.instrs.indexOf(it)
                block.instrs.add(idx, IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prims.Comp.PANIC),
                    mutableListOf()
                ))
            }
            else if (a.isPrim(it, Prims.Comp.ARR_STORE)) {
                if (it.args[0] == arr) {
                    val idx = block.instrs.indexOf(it)
                    block.instrs.add(idx, IrInstr(
                        mutableListOf(),
                        PrimitiveInstr(Prims.Comp.PANIC),
                        mutableListOf()
                    ))
                }
                else if (it.args[2] == arr) {
                    block.instrs.remove(it)
                }
            }
            else if (a.isPrim(it, Prims.Comp.DIM)) {
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
                        Prims.Comp.REPEAT -> 2
                        Prims.CALL -> 1
                        Prims.SWITCH -> 3
                        else -> return@forEach
                    }

                    a.function(fnRef.get())?.let { fn ->
                        val fnA = Analysis(fn)
                        if (debugVerify) {
                            if (fnA.callerInstrs().hasLeast(2))
                                error("you need to run oneblockonecaller before emptyArrayOpsRemove")
                        }

                        val arg = fn.args[argIdx]
                        log("tracing empty array from ${block.name} -> ${fn.name}")
                        emptyArray(fn, arg, putBlock)
                    }
                }
            }
        }
}

val emptyArrayOpsRemove = Pass<(IrBlock) -> Unit>("arr ops remove") { block, putBlock ->
    val a = Analysis(block)

    block.instrs.toList().forEach {
        if (a.isPrim(it, Prims.Comp.ARR_ALLOC)) {
            val arr = it.outs[0]
            val shapec = a.constShape(arr)
                ?: return@forEach

            if (shapec.none { it.a == 0 })
                return@forEach

            emptyArray(block, arr, putBlock)
        }
    }
}.withoutParallel()