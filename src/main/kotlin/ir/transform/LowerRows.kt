package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr

fun IrBlock.lowerRows(putBlock: (IrBlock) -> Unit ) {
    instrs.toList().forEach { instr ->
        if (instr.instr is PrimitiveInstr) {
            when (instr.instr.id) {
                Prim.ROWS -> {
                    var idx = instrs.indexOf(instr)
                    instrs.removeAt(idx)

                    val outType = instr.outs[0].type as ArrayType

                    val fn = instr.args[0]
                    val inputs = instr.args.drop(1)

                    val inputsLens = inputs.map {
                        val v = newVar().copy(type = Types.size)
                        instrs.add(idx ++, IrInstr(
                            mutableListOf(v),
                            PrimitiveInstr(Prim.LEN),
                            mutableListOf(it)
                        ))
                        v
                    }

                    val inputsLensArgArr = inputsLens.wrapInArgArray(::newVar) { instrs.add(idx ++, it) }
                    val inputsLensArr = newVar().copy(type = inputsLensArgArr.type)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(inputsLensArr),
                        PrimitiveInstr(Prim.Comp.ARR_MATERIALIZE),
                        mutableListOf(inputsLensArgArr),
                    ))

                    val maxInputsLen = newVar().copy(type = Types.size)

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
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(reduceBodyFn),
                        PushFnRefInstr(reduceBody.name),
                        mutableListOf()
                    ))

                    // TODO: wrap in fill(0)
                    instrs.add(idx ++, IrInstr(
                        mutableListOf(maxInputsLen),
                        PrimitiveInstr(Prim.REDUCE),
                        mutableListOf(reduceBodyFn, inputsLensArr, reduceBodyFn)
                    ))

                    // !!! to allocate the array, we first have to run the function once to get the dimension of the inner arrays

                    //   TODO: we should check if first iter even possible
                    //   execute iter 0
                    //   allocate array
                    //   copy result from iter 0 into array
                    //   free result from iter 0
                    //   repeat for remaining iters:
                    //     execute iter i
                    //     copy result from iter i into array
                    //     free result from iter i
                    //   ^ array

                    val iterFn = IrBlock(anonFnName(), ref, fillArg = fillArg).apply {
                        fillArg?.let { fillArg = newVar().copy(type = it.type) }

                        val idx = newVar().copy(type = Types.size).also { args += it }
                        val fn = newVar().copy(type = fn.type).also { args += it }
                        val maxInputsLen = newVar().copy(type = Types.size).also { args += it }
                        val inputs = inputs.map { newVar().copy(type = it.type).also { args += it } }

                        val inputsLens = inputs.map {
                            val v = newVar().copy(type = Types.size)
                            instrs += IrInstr(
                                mutableListOf(v),
                                PrimitiveInstr(Prim.LEN),
                                mutableListOf(it)
                            )
                            v
                        }

                        val output = newVar().copy(type = outType.of).also { rets += it }

                        // all inputs with exactly one row are reused for all rows of the other inputs

                        putBlock(this)
                    }
                }
            }
        }
    }
}