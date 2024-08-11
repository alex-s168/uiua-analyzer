package me.alex_s168.uiua.ir.opt

import blitz.Either
import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.analysisPass

val emptyArrayOpsRemove = analysisPass("infer array sizes") { block, a ->
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
            val arrType = arr.type as ArrayType

            if (shapec.none { it.isA && it.getA() == 0.0 })
                return@forEach

            a.usages(arr).forEach {
                it?.let {
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
                }
            }
        }
    }
}