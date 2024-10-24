package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.depth
import me.alex_s168.uiua.ir.transform.into

val lowerPervasive = withPassArg<(IrBlock) -> Unit>("lower pervasive") { putBlock ->
    optAwayPass(
        "lower pervasive",
        { it.instr is PrimitiveInstr && it.instr.id in Analysis.pervasive },
        { a -> outs[0].type is ArrayType },
    ) { put, newVar, a ->
        instr as PrimitiveInstr
        val outTy = outs[0].type as ArrayType

        depth(newVar, put, a.block.ref, putBlock, outTy.shape.size, args.toList()) { newVar, put, args ->
            val res = newVar().copy(type = outTy.inner)
            put(IrInstr(
                mutableListOf(res),
                PrimitiveInstr(instr.id),
                args.toMutableList()
            ))
            listOf(res)
        }[0].into(outs[0], put)
    }
}.parallelWithoutDeepCopy()