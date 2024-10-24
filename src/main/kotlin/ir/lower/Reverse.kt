package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.into
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerReverse = lowerPrimPass<(IrBlock) -> Unit>(Prim.REVERSE) { put, newVar, a, putBlock ->
    val arg = args[0]

    val operate = if (arg.type is BoxType) {
        val unboxed = newVar().copy(type = arg.type.of)
        put(IrInstr(
            mutableListOf(unboxed),
            PrimitiveInstr(Prim.UN_BOX),
            mutableListOf(arg)
        ))
        unboxed
    } else arg

    val operateTy = operate.type as ArrayType

    val out = if (arg.type is BoxType) newVar().copy(type = arg.type.of)
    else outs[0]
    val outTy = out.type

    val len = newVar().copy(type = Types.size)
    put(IrInstr(
        mutableListOf(len),
        PrimitiveInstr(Prim.LEN),
        mutableListOf(operate),
    ))

    val shape = listOf(len).wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(out),
        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
        mutableListOf(shape)
    ))

    val block = IrBlock(anonFnName(), a.block.ref).apply {
        val idx = newVar().copy(type = Types.size).also { args += it }
        val operate = newVar().copy(type = operateTy).also { args += it }
        val out = newVar().copy(type = outTy).also { args += it }
        val lenM1 = newVar().copy(type = Types.size).also { args += it }

        val indec = listOf(idx).wrapInArgArray(::newVar) { instrs += it }

        val temp = newVar().copy(type = operateTy.of.makeVaOffIfArray())
        instrs += IrInstr(
            mutableListOf(temp),
            PrimitiveInstr(Prim.Comp.ARR_LOAD),
            mutableListOf(operate, indec)
        )

        val revIdx = newVar().copy(type = Types.size)
        instrs += IrInstr(
            mutableListOf(revIdx),
            PrimitiveInstr(Prim.SUB),
            mutableListOf(idx, lenM1)
        )

        val revIdcs = listOf(revIdx).wrapInArgArray(::newVar) { instrs += it }

        instrs += IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prim.Comp.ARR_STORE),
            mutableListOf(out, revIdcs, temp)
        )

        putBlock(this)
    }

    val (zero, one) = constants(newVar, 0.0, 1.0, type = Types.size, put = put)

    val blockFn = newVar().copy(type = block.type())
    put(IrInstr(
        mutableListOf(blockFn),
        PushFnRefInstr(block.name),
        mutableListOf()
    ))

    val lenM1 = newVar().copy(type = Types.size)
    put(IrInstr(
        mutableListOf(lenM1),
        PrimitiveInstr(Prim.SUB),
        mutableListOf(one, len)
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.REPEAT),
        mutableListOf(zero, len, blockFn, operate, out, lenM1)
    ))

    out.into(outs[0], put = put)
}.parallelWithoutDeepCopy()