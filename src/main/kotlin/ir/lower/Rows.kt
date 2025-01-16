package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.*

val lowerRows = lowerPrimPass<(IrBlock) -> Unit>(Prims.ROWS) { put, newVar, a, putBlock ->
    val outTypes = outs.map { it.type as ArrayType }

    val fn = args[0]
    val inputs = args.drop(1)

    // the length of each output array (if none of the inputs are >1 rank arrays),
    // is identical to the maximum of all input arrays lengths
    val maxInputsLen = newVar().copy(type = Types.size)
    filled(Types.size, {
        instrs += IrInstr(
            mutableListOf(it),
            NumImmInstr(0.0),
            mutableListOf()
        )
    }, a.block.ref, newVar, putBlock, put, mutableListOf(maxInputsLen), *inputs.toTypedArray()) {
        val inputs = args
        val maxInputsLen = rets[0]

        fun put(it: IrInstr) {
            instrs.add(it)
        }

        val inputsLens = inputs.map {
            val v = newVar().copy(type = Types.size)
            put(IrInstr(
                mutableListOf(v),
                PrimitiveInstr(Prims.LEN),
                mutableListOf(it)
            ))
            v
        }

        val inputsLensArr = inputs.zip(inputsLens).filterNot { (it, _) ->
            it.type !is ArrayType || it.type.length == 1
        }.map { it.second }.wrapInArgArray(::newVar, Types.size, put = ::put)

        reduce(maxInputsLen, inputsLensArr, ::put, putBlock, a.block.ref, newVar, Types.size) { a, b, res ->
            instrs += IrInstr(
                mutableListOf(res),
                PrimitiveInstr(Prims.MAX),
                mutableListOf(a, b)
            )
        }
    }

    val empty = IrBlock(anonFnName(), a.block.ref, fillArg = a.block.fillArg).apply {
        fillArg?.let { fillArg = newVar().copy(type = it.type) }

        val fn = newVar().copy(type = fn.type).also { args += it }
        val maxInputsLen = newVar().copy(type = Types.size).also { args += it }
        val inputs = inputs.map { newVar().copy(type = it.type).also { args += it } }

        val outputs = outTypes.map { newVar().copy(type = it).also { rets += it } }

        outputs.forEach {
            val shape = constantArr(
                ::newVar,
                *(it.type as ArrayType).shape.map { 0.0 }.toDoubleArray(),
                type = Types.size
            ) { instrs += it }

            instrs += IrInstr(
                mutableListOf(it),
                PrimitiveInstr(Prims.Comp.ARR_ALLOC),
                mutableListOf(shape)
            )
        }

        putBlock(this)
    }

    val full = IrBlock(anonFnName(), a.block.ref, fillArg = a.block.fillArg).apply {
        // !!! to allocate the array, we first have to run the function once to get the dimension of the inner arrays
        // (not always tho)
        //   execute iter 0
        //   allocate array
        //   copy result from iter 0 into array
        //   free result from iter 0
        //   repeat for remaining iters:
        //     execute iter i
        //     copy result from iter i into array
        //     free result from iter i
        //   ^ array

        fillArg?.let { fillArg = newVar().copy(type = it.type) }

        val fn = newVar().copy(type = fn.type).also { args += it }
        val maxInputsLen = newVar().copy(type = Types.size).also { args += it }
        val inputs = inputs.map { newVar().copy(type = it.type).also { args += it } }

        val outputs = outTypes.map { newVar().copy(type = it).also { rets += it } }

        val (zero, one) = constants(::newVar, 0.0, 1.0, type = Types.size) { instrs += it }

        val (startAt, shapes, toStoreIter0) = if (outTypes.any { it.of is ArrayType && it.of.shape.any { it == -1 } }) {
            // do iter 0 outside of loop

            val loadedInputs0 = inputs.map {
                if (it.type !is ArrayType || it.type.length == 1) {
                    it
                } else {
                    oneDimFillLoad(fillArg, it, putBlock, ref, ::newVar, zero) { instrs += it }
                }
            }

            val iter0 = outTypes.map { newVar().copy(type = it.of) }
            instrs += IrInstr(
                iter0.toMutableList(),
                PrimitiveInstr(Prims.CALL),
                mutableListOf(fn).also { it += loadedInputs0 }
            )

            Triple(
                one,
                iter0.map { arr ->
                    val shape = mutableListOf(maxInputsLen)
                    if (arr.type is ArrayType) {
                        arr.type.shape.forEachIndexed { index, _ ->
                            val (x) = constants(::newVar, index.toDouble(), type = Types.size) { instrs += it }

                            val v = newVar().copy(type = Types.size)
                            instrs += IrInstr(
                                mutableListOf(v),
                                PrimitiveInstr(Prims.Comp.DIM),
                                mutableListOf(arr, x)
                            )
                            shape += v
                        }
                    }
                    shape.wrapInArgArray(::newVar) { instrs += it }
                },
                iter0
            )
        } else Triple(
            zero,
            outTypes.map { ot ->
                val shape = mutableListOf(maxInputsLen)
                if (ot.of is ArrayType) {
                    ot.of.shape.forEach {
                        assert(it != -1)
                        shape += constant(it.toDouble(), Types.size, ::newVar, instrs::add)
                    }
                }
                shape.wrapInArgArray(::newVar) { instrs += it }
            },
            null
        )

        outputs.zip(shapes).forEach { (arr, sha) ->
            instrs += IrInstr(
                mutableListOf(arr),
                PrimitiveInstr(Prims.Comp.ARR_ALLOC),
                mutableListOf(sha)
            )
        }

        val indc0 = listOf(zero).wrapInArgArray(::newVar) { instrs += it }

        toStoreIter0?.zip(outputs)?.forEach { (src, dest) ->
            instrs += IrInstr(
                mutableListOf(),
                PrimitiveInstr(Prims.Comp.ARR_STORE),
                mutableListOf(dest, indc0, src)
            )
        }

        val iterFnLoop = IrBlock(anonFnName(), ref, fillArg = fillArg).apply {
            fillArg?.let { fillArg = newVar().copy(type = it.type) }

            val idx = newVar().copy(type = Types.size).also { args += it }
            val fn = newVar().copy(type = fn.type).also { args += it }
            val inputs = inputs.map { newVar().copy(type = it.type).also { args += it } }
            val outputs = outTypes.map { newVar().copy(type = it).also { args += it } }

            // all inputs with exactly one row are reused for all rows of the other inputs
            // this requires the compiler to know at comptime that 1 row
            val loadedInputs = inputs.map {
                if (it.type !is ArrayType || it.type.length == 1) {
                    it
                } else {
                    oneDimFillLoad(fillArg, it, putBlock, ref, ::newVar, idx) { instrs += it }
                }
            }

            val iter = outTypes.map { newVar().copy(type = it.of).also { rets += it } }
            instrs += IrInstr(
                iter.toMutableList(),
                PrimitiveInstr(Prims.CALL),
                mutableListOf(fn).also { it += loadedInputs }
            )

            val indc = listOf(idx).wrapInArgArray(::newVar) { instrs += it }

            iter.zip(outputs).forEach { (src, destArr) ->
                instrs += IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prims.Comp.ARR_STORE),
                    mutableListOf(destArr, indc, src)
                )
            }

            putBlock(this)
        }

        val iterFnLoopFn = newVar().copy(type = iterFnLoop.type())
        instrs += IrInstr(
            mutableListOf(iterFnLoopFn),
            PushFnRefInstr(iterFnLoop.uid),
            mutableListOf()
        )

        instrs += IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prims.Comp.REPEAT),
            mutableListOf(startAt, maxInputsLen, iterFnLoopFn, fn).also {
                it += inputs
                it += outputs
            }
        )

        putBlock(this)
    }

    val (zero, one) = constants(newVar, 0.0, 1.0, type = Types.size, put = put)

    switch(
        outs,
        newVar,
        maxInputsLen,
        mutableListOf(fn, maxInputsLen).also { it += inputs },
        zero to empty,
        one to full,
        put = put
    )
}
