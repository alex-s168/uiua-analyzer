package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.reduce
import me.alex_s168.uiua.ir.transform.switch
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerReshape = withPassArg<(IrBlock) -> Unit>("lower reshape") { putBlock ->
    lowerPrimPass(Prims.RESHAPE) { put, newVar, a ->
        if (args[0].type !is ArrayType) { // => scalar
            // repeat array as rows of new array
            val oldValue = args[1]
            val oldValueType = oldValue.type
            val out = outs[0]

            val shape = mutableListOf(args[0])
            if (oldValueType is ArrayType) {
                shape += List(oldValueType.shape.size) {
                    val (d) = constants(newVar, it.toDouble(), type = Types.size, put = put)

                    val dim = newVar().copy(type = Types.size)
                    put(IrInstr(
                        mutableListOf(dim),
                        PrimitiveInstr(Prims.Comp.DIM),
                        mutableListOf(oldValue, d)
                    ))
                    dim
                }
            }

            put(IrInstr(
                mutableListOf(out),
                PrimitiveInstr(Prims.Comp.ARR_ALLOC),
                mutableListOf(shape.wrapInArgArray(newVar, put = put))
            ))

            val loopBody = IrBlock(anonFnName(), a.block.ref).apply {
                val iteration = newVar().copy(type = Types.size).also(args::add)
                val out = newVar().copy(type = out.type).also(args::add)
                val oldValue = newVar().copy(type = oldValueType).also(args::add)

                val idx = listOf(iteration)
                    .wrapInArgArray(newVar, put = instrs::add)

                instrs += IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prims.Comp.ARR_STORE),
                    mutableListOf(out, idx, oldValue)
                )

                putBlock(this)
            }

            val fnref = newVar().copy(type = loopBody.type())
            put(IrInstr(
                mutableListOf(fnref),
                PushFnRefInstr(loopBody.name),
                mutableListOf()
            ))

            val (zero) = constants(newVar, 0.0, type = Types.size, put = put)

            put(IrInstr(
                mutableListOf(),
                PrimitiveInstr(Prims.Comp.REPEAT),
                mutableListOf(zero, args[0], fnref, out, oldValue)
            ))
        }
        else {
            val newShape = args[0]
            // val newShapeV = a.origin(args[0])!!.args.toList()
            val oldValue = args[1]

            val matNewShape = newVar().copy(type = newShape.type)
            put(IrInstr(
                mutableListOf(matNewShape),
                PrimitiveInstr(Prims.Comp.ARR_MATERIALIZE),
                mutableListOf(newShape)
            ))

            val des = newVar().copy(type = Types.array((oldValue.type as ArrayType).inner))
            put(IrInstr(
                mutableListOf(des),
                PrimitiveInstr(Prims.DESHAPE),
                mutableListOf(oldValue)
            ))

            val newShapeMul = reduce(matNewShape, put, putBlock, a.block.ref, newVar, Types.size) { a, b, res ->
                instrs += IrInstr(
                    mutableListOf(res),
                    PrimitiveInstr(Prims.MUL),
                    mutableListOf(a, b)
                )
            }

            val desLen = newVar().copy(type = Types.size)
            put(IrInstr(
                mutableListOf(desLen),
                PrimitiveInstr(Prims.LEN),
                mutableListOf(des)
            ))

            // if desLen > newShapeMul
            //  truncate
            //
            // if desLen < newShapeMul
            //  if have fill value:
            //   use fill value to extend
            //  else:
            //   cycle trough old values

            val bool = newVar().copy(type = Types.bool)
            put(IrInstr(
                mutableListOf(bool),
                PrimitiveInstr(Prims.LT),
                mutableListOf(desLen, newShapeMul)
            ))

            // also gets executed if equal sizes
            val blkTrunc = IrBlock(anonFnName(), a.block.ref).apply {
                fillArg = a.block.fillArg?.let { newVar().copy(type = it.type) }
                val newShapeMul = newVar().copy(type = newShapeMul.type).also(args::add)
                val desLen = newVar().copy(type = desLen.type).also(args::add)
                val des = newVar().copy(type = des.type).also(args::add)

                val correctSizeArr = newVar().copy(type = des.type).also(rets::add)

                // allocating here because in blkExtend a new array is allocated too
                instrs += IrInstr(
                    mutableListOf(correctSizeArr),
                    PrimitiveInstr(Prims.Comp.ARR_CLONE),
                    mutableListOf(des)
                )

                putBlock(this)
            }

            val blkExtend = IrBlock(anonFnName(), a.block.ref).apply {
                fillArg = a.block.fillArg?.let { newVar().copy(type = it.type) }
                val newShapeMul = newVar().copy(type = newShapeMul.type).also(args::add)
                val desLen = newVar().copy(type = desLen.type).also(args::add)
                val des = newVar().copy(type = des.type).also(args::add)

                val correctSizeArr = newVar().copy(type = des.type).also(rets::add)

                if (fillArg != null) {
                    instrs += IrInstr(
                        mutableListOf(correctSizeArr),
                        PrimitiveInstr(Prims.Comp.RT_EXTEND_SCALAR), // TODO: move reshape<scalar> into own primitive and use that + some arr copies
                        mutableListOf(des, newShapeMul, fillArg!!)
                    )
                }
                else {
                    instrs += IrInstr(
                        mutableListOf(correctSizeArr),
                        PrimitiveInstr(Prims.Comp.RT_EXTEND_REPEAT),
                        mutableListOf(des, newShapeMul, des)
                    )
                }

                putBlock(this)
            }

            // des is a copy of the input array already, no need for extra copy

            val correctSizeArr = newVar().copy(type = des.type)

            val (zero, one) = constants(newVar, 0.0, 1.0, type = Types.bool, put = put)
            switch(
                dest = listOf(correctSizeArr),
                newVar = newVar,
                on = bool,
                inputs = listOf(newShapeMul, desLen, des),
                zero to blkTrunc,
                one to blkExtend,
                put = put,
            )

            put(IrInstr(
                mutableListOf(outs[0]),
                PrimitiveInstr(Prims.Comp.RESHAPE_VIEW),
                mutableListOf(newShape, correctSizeArr)
            ))
        }
    }
}.parallelWithoutDeepCopy()