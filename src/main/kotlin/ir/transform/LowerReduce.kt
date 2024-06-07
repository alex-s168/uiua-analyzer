package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerReduce(putBlock: (IrBlock) -> Unit) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.REDUCE -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val accTy = instr.outs[0].type

                    val arr = instr.args[1]
                    val arrTy = arr.type as ArrayType

                    val first = instr.args[0]
                    val firstTy = first.type as FnType

                    val extra = instr.args.subList(2, instr.args.size - 1)

                    val every = instr.args.last()
                    val everyTy = every.type as FnType

                    val notEnoughElemsFn = IrBlock(anonFnName(), ref, fillArg = fillArg).apply {
                        fillArg?.let { fillArg = newVar().copy(type = it.type) }

                        args += newVar().copy(type = arrTy) // arr
                        args += newVar().copy(type = firstTy) // first
                        args += newVar().copy(type = everyTy) // every
                        extra.forEach {
                            args += newVar().copy(type = it.type)
                        }

                        val default = fillArg ?: if (accTy is NumericType) {
                            val v = newVar().copy(type = accTy)
                            instrs += IrInstr(
                                mutableListOf(v),
                                NumImmInstr(0.0),
                                mutableListOf()
                            )
                            v
                        } else {
                            val v = newVar().copy(type = accTy)
                            instrs += IrInstr(
                                mutableListOf(v),
                                PrimitiveInstr(Prim.Comp.PANIC),
                                mutableListOf()
                            )
                            v
                        }

                        rets += default

                        putBlock(this)
                    }

                    val enoughElemsFn = IrBlock(anonFnName(), ref).apply {
                        val arr = newVar().copy(type = arrTy).also { args += it }
                        val first = newVar().copy(type = firstTy).also { args += it }
                        val every = newVar().copy(type = everyTy).also { args += it }
                        val extra = extra.map {
                            newVar().copy(type = it.type).also { args += it }
                        }

                        val elem0 = oneDimLoad(arr, ::newVar, 0) { instrs += it }
                        val elem1 = oneDimLoad(arr, ::newVar, 1) { instrs += it }

                        val iter0res = newVar().copy(type = accTy)
                        instrs += IrInstr(
                            mutableListOf(iter0res),
                            PrimitiveInstr(Prim.CALL),
                            mutableListOf(first, elem0, elem1).also { it += extra }
                        )

                        val accBox = newVar().copy(type = Types.box(accTy))
                        instrs += IrInstr(
                            mutableListOf(accBox),
                            PrimitiveInstr(Prim.BOX),
                            mutableListOf(iter0res)
                        )

                        val len = newVar().copy(type = Types.int)
                        instrs += IrInstr(
                            mutableListOf(len),
                            PrimitiveInstr(Prim.LEN),
                            mutableListOf(arr)
                        )

                        val start = newVar().copy(type = Types.int)
                        instrs += IrInstr(
                            mutableListOf(start),
                            NumImmInstr(2.0),
                            mutableListOf()
                        )

                        val iterFn = IrBlock(anonFnName(), ref).apply {
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
                        instrs += IrInstr(
                            mutableListOf(iterFnRef),
                            PushFnRefInstr(iterFn.name),
                            mutableListOf(),
                        )

                        instrs += IrInstr(
                            mutableListOf(),
                            PrimitiveInstr(Prim.Comp.REPEAT), // [start], [end], [fn which takes counter], [additional]...
                            mutableListOf(start, len, iterFnRef, arr, accBox, every).also { it += extra }
                        )

                        val acc = newVar().copy(type = accTy)
                        instrs += IrInstr(
                            mutableListOf(acc),
                            PrimitiveInstr(Prim.UN_BOX),
                            mutableListOf(accBox)
                        )

                        rets += acc

                        putBlock(this)
                    }

                    val len = newVar().copy(type = Types.int)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(len),
                        PrimitiveInstr(Prim.LEN),
                        mutableListOf(arr)
                    ))

                    // args to fns:
                    //  arr
                    //  first
                    //  every
                    //  ...extra

                    // <notEnough | notEnough | enough> len

                    val (zero, one, two) = constants(::newVar, 0.0, 1.0, 2.0, type = Types.int) {
                        instrs.add(idx ++, it)
                    }

                    switch(
                        dest = instr.outs,
                        newVar = ::newVar,
                        on = len,
                        inputs = listOf(arr, first, every) + extra,
                        zero to notEnoughElemsFn,
                        one to notEnoughElemsFn,
                        two to enoughElemsFn,
                    ) { instrs.add(idx ++, it) }
                }
            }
        }
    }
}