package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.depth
import me.alex_s168.uiua.ir.transform.into

val lowerReduceDepth = withPassArg<(IrBlock) -> Unit>("lower reduce depth") { putBlock ->
    lowerPrimPass(Prims.Front.REDUCE_DEPTH) { put, newVar, a ->
        val depth = (instr as PrimitiveInstr).param!!

        val oldOut = outs.toList()

        depth(newVar, put, a.block.ref, putBlock, depth, args) { newVar, put, args ->
            val out = oldOut.map { newVar().copy(type = it.type.ofIfArray()) }
            put(IrInstr(
                out.toMutableList(),
                PrimitiveInstr(Prims.REDUCE),
                args.toMutableList()
            ))
            out
        }.zip(oldOut).forEach { (new, old) ->
            new.into(old, put)
        }
    }
}.parallelWithoutDeepCopy()