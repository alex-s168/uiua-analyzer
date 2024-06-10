package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerRows(putBlock: (IrBlock) -> Unit ) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.ROWS -> {
                    rows(instr, putBlock)
                }
            }
        }
    }
}

private fun IrBlock.rows(instr: IrInstr, putBlock: (IrBlock) -> Unit) {
    var idx = instrs.indexOf(instr)
    instrs.removeAt(idx)

    instrs.add(idx ++, comment("+++ $instr"))

    val outTypes = instr.outs.map { it.type as ArrayType }

    val fn = instr.args[0]
    val inputs = instr.args.drop(1)

    val maxInputsLen = newVar().copy(type = Types.size)

    filled(Types.size, {
        instrs += IrInstr(
            mutableListOf(it),
            NumImmInstr(0.0),
            mutableListOf()
        )
    }, ref, ::newVar, putBlock, {
        instrs.add(idx ++, it)
    }, mutableListOf(maxInputsLen), *inputs.toTypedArray()) {
        val inputs = args
        val maxInputsLen = rets[0]

        val inputsLens = inputs.map {
            val v = newVar().copy(type = Types.size)
            instrs.add(IrInstr(
                mutableListOf(v),
                PrimitiveInstr(Prim.LEN),
                mutableListOf(it)
            ))
            v
        }

        val inputsLensArgArr = inputs.zip(inputsLens).filterNot { (it, _) ->
            it.type !is ArrayType || it.type.length == 1
        }.map { it.second }.wrapInArgArray(::newVar, Types.size) { instrs += it }
        val inputsLensArr = newVar().copy(type = inputsLensArgArr.type)
        instrs.add(IrInstr(
            mutableListOf(inputsLensArr),
            PrimitiveInstr(Prim.Comp.ARR_MATERIALIZE),
            mutableListOf(inputsLensArgArr),
        ))

        val reduceBody = IrBlock(anonFnName(), ref).apply {
            val a = newVar().copy(type = Types.size).also { args += it }
            val b = newVar().copy(type = Types.size).also { args += it }

            val res = newVar().copy(type = Types.size).also { rets += it }

            instrs += IrInstr(
                mutableListOf(res),
                PrimitiveInstr(Prim.MAX),
                mutableListOf(a, b)
            )

            putBlock(this)
        }
        val reduceBodyFn = newVar().copy(type = reduceBody.type())
        instrs.add(IrInstr(
            mutableListOf(reduceBodyFn),
            PushFnRefInstr(reduceBody.name),
            mutableListOf()
        ))

        instrs.add(IrInstr(
            mutableListOf(maxInputsLen),
            PrimitiveInstr(Prim.REDUCE),
            mutableListOf(reduceBodyFn, inputsLensArr, reduceBodyFn)
        ))
    }

    // !!! to allocate the array, we first have to run the function once to get the dimension of the inner arrays

    //   execute iter 0
    //   allocate array
    //   copy result from iter 0 into array
    //   free result from iter 0
    //   repeat for remaining iters:
    //     execute iter i
    //     copy result from iter i into array
    //     free result from iter i
    //   ^ array

    val empty = IrBlock(anonFnName(), ref, fillArg = fillArg).apply {
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
                PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                mutableListOf(shape)
            )
        }

        putBlock(this)
    }

    val full = IrBlock(anonFnName(), ref, fillArg = fillArg).apply {
        fillArg?.let { fillArg = newVar().copy(type = it.type) }

        val fn = newVar().copy(type = fn.type).also { args += it }
        val maxInputsLen = newVar().copy(type = Types.size).also { args += it }
        val inputs = inputs.map { newVar().copy(type = it.type).also { args += it } }

        val outputs = outTypes.map { newVar().copy(type = it).also { rets += it } }

        val (zero, one) = constants(::newVar, 0.0, 1.0, type = Types.size) { instrs += it }

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
            PrimitiveInstr(Prim.CALL),
            mutableListOf(fn).also { it += loadedInputs0 }
        )

        val shapes = iter0.map { arr ->
            val shape = mutableListOf(maxInputsLen)
            if (arr.type is ArrayType) {
                arr.type.shape.forEachIndexed { index, _ ->
                    val (x) = constants(::newVar, index.toDouble(), type = Types.size) { instrs += it }

                    val v = newVar().copy(type = Types.size)
                    instrs += IrInstr(
                        mutableListOf(v),
                        PrimitiveInstr(Prim.Comp.DIM),
                        mutableListOf(arr, x)
                    )
                    shape += v
                }
            }
            shape.wrapInArgArray(::newVar) { instrs += it }
        }

        outputs.zip(shapes).forEach { (arr, sha) ->
            instrs += IrInstr(
                mutableListOf(arr),
                PrimitiveInstr(Prim.Comp.ARR_ALLOC),
                mutableListOf(sha)
            )
        }

        val indc0 = listOf(zero).wrapInArgArray(::newVar) { instrs += it }

        iter0.zip(outputs).forEach { (src, dest) ->
            instrs += IrInstr(
                mutableListOf(),
                PrimitiveInstr(Prim.Comp.ARR_STORE),
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
                PrimitiveInstr(Prim.CALL),
                mutableListOf(fn).also { it += loadedInputs }
            )

            val indc = listOf(idx).wrapInArgArray(::newVar) { instrs += it }

            iter.zip(outputs).forEach { (src, destArr) ->
                instrs += IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prim.Comp.ARR_STORE),
                    mutableListOf(destArr, indc, src)
                )
            }

            putBlock(this)
        }

        val iterFnLoopFn = newVar().copy(type = iterFnLoop.type())
        instrs += IrInstr(
            mutableListOf(iterFnLoopFn),
            PushFnRefInstr(iterFnLoop.name),
            mutableListOf()
        )

        instrs += IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prim.Comp.REPEAT),
            mutableListOf(one, maxInputsLen, iterFnLoopFn, fn).also {
                it += inputs
                it += outputs
            }
        )

        putBlock(this)
    }

    val (zero, one) = constants(::newVar, 0.0, 1.0, type = Types.size) {
        instrs.add(idx ++, it)
    }

    switch(
        instr.outs,
        ::newVar,
        maxInputsLen,
        mutableListOf(fn, maxInputsLen).also { it += inputs },
        zero to empty,
        one to full,
    ) { instrs.add(idx ++, it) }

    instrs.add(idx ++, comment("--- $instr"))
}