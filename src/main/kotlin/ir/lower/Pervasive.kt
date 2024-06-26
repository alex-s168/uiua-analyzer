package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.optAwayPass
import me.alex_s168.uiua.ir.withPassArg

val lowerPervasive = withPassArg<(IrBlock) -> Unit>("lower pervasive") { putBlock ->
    optAwayPass(
        "lower pervasive",
        { it.instr is PrimitiveInstr && it.instr.id in Analysis.pervasive },
        { a -> outs[0].type is ArrayType },
    ) { put, newVar, a ->
        instr as PrimitiveInstr
        val outTy = outs[0].type as ArrayType

        val aTy = args[0].type.let { if (it is ArrayType) it.inner else it }
        val bTy = args[1].type.let { if (it is ArrayType) it.inner else it }

        val fn = IrBlock(anonFnName(), a.block.ref).apply {
            val a = newVar().copy(type = aTy).also { args += it }
            val b = newVar().copy(type = bTy).also { args += it }

            val res = newVar().copy(type = outTy.inner).also { rets += it }

            instrs += IrInstr(
                mutableListOf(res),
                PrimitiveInstr(instr.id),
                mutableListOf(a, b)
            )

            putBlock(this)
        }

        val fnFn = newVar().copy(type = fn.type())
        put(IrInstr(
            mutableListOf(fnFn),
            PushFnRefInstr(fn.name),
            mutableListOf()
        ))

        put(IrInstr(
            mutableListOf(outs[0]),
            PrimitiveInstr(Prim.EACH),
            mutableListOf(fnFn, args[0], args[1])
        ))
    }
}