package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.genCallBlockFnTail
import me.alex_s168.uiua.ir.transform.genFix

val lowerTable = lowerPrimPass<(IrBlock) -> Unit>(Prim.TABLE) { put, newVar, a, putBlock ->
    val allOuts = outs
    val argFn = args[0]
    val argArrays = args.drop(1)

    val fixedArrays = argArrays.mapIndexed { idx, irVar -> List(idx){it}
        .fold(irVar) { v, _ -> v.genFix(put, newVar) } }

    val inner = genCallBlockFnTail(argFn.type as FnType, a.block.ref)
        .also(putBlock)

    val outer = List(argArrays.size){it}
        .foldRight(inner) { depth, block ->
            IrBlock(anonFnName(), a.block.ref).apply {
                val ar = argArrays.take(depth + 1)
                    .map { newVar().copy(type = it.type)
                        .also { args += it } }

                val fn = newVar().copy(type = argFn.type)
                    .also { args += it }

                val outs = allOuts
                    .map { newVar().copy(type = it.type)
                        .also { rets += it } }

                instrs += IrInstr(
                    outs.toMutableList(),
                    PrimitiveInstr(Prim.ROWS),
                    (listOf(fn) + ar).toMutableList()
                )
            }.also(putBlock)
        }

    val fnref = newVar().copy(type = outer.type())

    put(IrInstr(
        mutableListOf(fnref),
        PushFnRefInstr(outer.name),
        mutableListOf()
    ))

    put(IrInstr(
        allOuts.toMutableList(),
        PrimitiveInstr(Prim.CALL),
        mutableListOf(fnref)
            .also { it += fixedArrays }
            .also { it += argFn }
    ))
}