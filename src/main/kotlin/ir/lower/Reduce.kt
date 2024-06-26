package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.*

val lowerReduce = lowerPrimPass<(IrBlock) -> Unit>(Prim.REDUCE) { put, newVar, a, putBlock ->
    val accTy = outs[0].type

    val arr = args[1]
    val arrTy = arr.type as ArrayType

    val first = args[0]
    val firstTy = first.type as FnType

    val extra = args.subList(2, args.size - 1)

    val every = args.last()
    val everyTy = every.type as FnType

    val len = newVar().copy(type = Types.size)
    put(IrInstr(
        mutableListOf(len),
        PrimitiveInstr(Prim.LEN),
        mutableListOf(arr)
    ))

    val (zero, one, two) = constants(newVar, 0.0, 1.0, 2.0, type = Types.int, put = put)

    fun buildLenFn(len: Int) =
        IrBlock(anonFnName(), a.block.ref, fillArg = a.block.fillArg).apply {
            fillArg?.let { fillArg = newVar().copy(type = it.type) }

            val arr = newVar().copy(type = arrTy).also { args += it }

            val res0 = newVar().copy(type = arrTy.of).also { rets += it }
            val res1 = newVar().copy(type = arrTy.of).also { rets += it }

            if (len < 2) {
                if (fillArg == null) {
                    instrs += IrInstr(
                        mutableListOf(res0, res1),
                        PrimitiveInstr(Prim.Comp.PANIC),
                        mutableListOf()
                    )
                } else if (len == 0) {
                    fillArg!!.into(res0, instrs::add)
                    fillArg!!.into(res1, instrs::add)
                }
                else if (len == 1) {
                    oneDimLoad(res0, arr, ::newVar, 0) { instrs += it }
                    fillArg!!.into(res1, instrs::add)
                }
            }
            else {
                oneDimLoad(res0, arr, ::newVar, 0) { instrs += it }
                oneDimLoad(res1, arr, ::newVar, 1) { instrs += it }
            }

            putBlock(this)
        }

    val elem0 = newVar().copy(type = arrTy.of)
    val elem1 = newVar().copy(type = arrTy.of)

    switch(
        dest = listOf(elem0, elem1),
        newVar = newVar,
        on = len,
        inputs = listOf(arr),
        zero to buildLenFn(0),
        one to buildLenFn(1),
        two to buildLenFn(2),
        put = put
    )

    val iter0res = newVar().copy(type = accTy)
    put(IrInstr(
        mutableListOf(iter0res),
        PrimitiveInstr(Prim.CALL),
        mutableListOf(first, elem0, elem1).also { it += extra }
    ))

    val accBox = newVar().copy(type = Types.box(accTy))
    put(IrInstr(
        mutableListOf(accBox),
        PrimitiveInstr(Prim.BOX),
        mutableListOf(iter0res)
    ))

    val iterFn = IrBlock(anonFnName(), a.block.ref).apply {
        val counter = newVar().copy(type = Types.int).also { args += it }
        val arr = newVar().copy(type = arrTy).also { args += it }
        val accBox = newVar().copy(type = Types.box(accTy)).also { args += it }
        val every = newVar().copy(type = everyTy).also { args += it }
        val extra = extra.map {
            newVar().copy(type = it.type).also { args += it }
        }

        val acc = newVar().copy(type = accTy)
        instrs += IrInstr(
            mutableListOf(acc),
            PrimitiveInstr(Prim.UN_BOX),
            mutableListOf(accBox)
        )

        val elem = oneDimLoad(arr, ::newVar, counter) { instrs += it }

        val newAcc = newVar().copy(type = accTy)
        instrs += IrInstr(
            mutableListOf(newAcc),
            PrimitiveInstr(Prim.CALL),
            mutableListOf(every, acc, elem).also { it += extra }
        )

        instrs += IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prim.Comp.BOX_STORE),
            mutableListOf(accBox, newAcc)
        )

        putBlock(this)
    }

    val iterFnRef = newVar().copy(type = iterFn.type())
    put(IrInstr(
        mutableListOf(iterFnRef),
        PushFnRefInstr(iterFn.name),
        mutableListOf(),
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.REPEAT), // [start], [end], [fn which takes counter], [additional]...
        mutableListOf(two, len, iterFnRef, arr, accBox, every).also { it += extra }
    ))

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.UN_BOX),
        mutableListOf(accBox)
    ))
}