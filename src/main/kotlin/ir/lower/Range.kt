package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerRange = lowerPrimPass<(IrBlock) -> Unit>(Prim.RANGE) { put, newVar, a, putBlock ->
    val arrTy = outs[0].type as ArrayType

    val shape = args.wrapInArgArray(newVar, put = put)

    put(IrInstr(
        outs,
        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
        mutableListOf(shape)
    ))

    val (zero) = constants(newVar, 0.0, type = Types.int, put = put)

    val fn = IrBlock(anonFnName(), a.block.ref, fillArg = a.block.fillArg).apply {
        fillArg?.let { fillArg = newVar().copy(type = it.type) }

        val index = newVar().copy(type = Types.size).also { args += it }
        val out = newVar().copy(type = arrTy).also { args += it }

        val indecies = listOf(index).wrapInArgArray(::newVar, put = instrs::add)

        instrs += IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prim.Comp.ARR_STORE),
            mutableListOf(out, indecies, index)
        )

        putBlock(this)
    }

    val fnRef = newVar().copy(type = fn.type())
    put(IrInstr(
        mutableListOf(fnRef),
        PushFnRefInstr(fn.name),
        mutableListOf()
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.REPEAT), // [start], [end], [fn which takes counter], [additional]...
        mutableListOf(zero, args[0], fnRef, outs[0])
    ))
}.parallelWithoutDeepCopy()