package me.alex_s168.uiua.ir

import blitz.cast
import blitz.flatten
import me.alex_s168.uiua.*
import kotlin.math.floor
import kotlin.math.max

fun IrInstr.inferTypes(
    parent: IrBlock,
    putFn: (IrBlock) -> Unit,
    fillType: Type? = null,
    verify: Boolean = false,
) {
    val a = Analysis(parent)
    // TODO: cache function() and deepOrigin here
    
    fun updateType(variable: IrVar, type: Type) {
        parent.updateVar(variable, variable.copy(type = type))
    }

    fun fnRef(to: BlockId): IrVar {
        val fn = parent.ref[to]!!
        val newv = parent.newVar().copy(type = fn.type())
        parent.instrs.add(
            parent.instrs.indexOf(this), IrInstr(
                mutableListOf(newv),
                PushFnRefInstr(to),
                mutableListOf(),
            )
        )
        return newv
    }

    fun highestShapeType(a: ArrayType, b: ArrayType): ArrayType {
        val asl = a.shape.size
        val bsl = b.shape.size

        if (asl > bsl)
            return a

        if (bsl > asl)
            return b

        return a.combineShape(b)
    }

    fun highestShapeType(a: Type, b: Type) =
        when (a) {
            is ArrayType -> when (b) {
                is ArrayType -> highestShapeType(a, b)
                else ->  a
            }
            else -> b
        }

    // TODO: IMPORTANT!!!!! use everywhere here
    fun expandArgFn(arg: Int, inTypes: List<Type>, fillType: Type? = null): IrBlock {
        if (verify) error("wtf")
        val pushfn = parent.instrDeclFor(args[arg])!!
        val fn = parent.ref[(pushfn.instr as PushFnRefInstr).fn]!!
        val exp = fn.expandFor(inTypes, putFn, fillType)
        pushfn.instr.fn = exp
        val expb = parent.ref[exp]!!
        updateType(pushfn.outs[0], expb.type())
        return expb
    }

    when (instr) {
        is ArrImmInstr -> {
            updateType(outs[0], instr.type.copy(vaOff = true, length = instr.values.flatten().size))
        }

        is NumImmInstr -> {
            if (!verify) {
                val ty = if (floor(instr.value) == instr.value) {
                    if (instr.value in 0.0..255.0) Types.autobyte
                    else Types.int
                } else Types.double
                updateType(outs[0], ty)
            }
        }

        is PushFnRefInstr -> {
            val fn = parent.ref[instr.fn]!!
            updateType(outs[0], fn.type())
        }

        is PrimitiveInstr -> when (instr.id) {
            Prims.Comp.USE -> {
                updateType(outs[0], args[0].type)
            }

            Prims.Comp.BOX_LOAD -> {
                val box = args[0]
                val boxType = box.type as BoxType
                updateType(outs[0], boxType.of)
            }

            Prims.Comp.BOX_STORE -> {
                val box = args[0]
                val boxType = box.type as BoxType
                val value = args[1]
                require(value.type == boxType.of)
            }

            Prims.Comp.BOX_CREATE -> {
                val box = outs[0]
                val boxType = box.type as BoxType

                if (boxType == Types.tbd) {
                    val usage = parent.instrs.firstOrNull {
                        it.instr is PrimitiveInstr &&
                                it.instr.id == Prims.Comp.BOX_STORE &&
                                box in it.args
                    } ?: error("untyped box not allowed!")

                    val storedVal = usage.args[1]

                    updateType(box, Types.box(storedVal.type))
                }
            }

            Prims.Comp.BOX_DESTROY -> {}

            Prims.Comp.ARG_ARR -> {
                updateType(outs[0], Types.array(args[0].type, args.size))
            }

            Prims.Comp.ARR_ALLOC -> {
                require(outs[0].type != Types.tbd)
            }

            Prims.Comp.ARR_STORE -> {}

            Prims.Comp.ARR_LOAD -> {
                val arrType = args[0].type as ArrayType
                val indecies = parent.instrDeclFor(args[1])!!
                val elemType = arrType.into(indecies.args.size)
                updateType(outs[0], elemType)
            }

            Prims.Comp.ARR_DESTROY -> {}

            Prims.Comp.REPEAT -> {}

            Prims.Comp.DIM -> {
                updateType(outs[0], Types.size)
            }

            Prims.DESHAPE -> {
                val arrTy = args[0].type as ArrayType
                updateType(outs[0], Types.array(arrTy.inner))
            }

            Prims.CALL -> {
                val callArgs = args.drop(1)
                if (verify) {
                    a.function(args[0])!!.type().let {
                        require(it.args.tyEq(callArgs.map(IrVar::type)))
                        require(it.rets.tyEq(outs.map(IrVar::type)))
                        require(it.fillType == fillType)
                    }
                } else {
                    val expb = expandArgFn(0, callArgs.map { it.type }, fillType)
                    outs.zip(expb.rets).forEach { (a, b) ->
                        updateType(a, b.type)
                    }
                }
            }

            Prims.BOX -> {
                updateType(outs[0], Types.box(args[0].type))
            }

            Prims.UN_BOX -> {
                val ty = args[0].type as BoxType
                updateType(outs[0], ty.of)
            }

            Prims.POW,
            Prims.DIV -> {
                val arrTy = args.firstOrNull { it.type is ArrayType }?.type as ArrayType?
                updateType(outs[0], arrTy?.mapInner { Types.double } ?: Types.double)
            }

            Prims.ABS,
            Prims.NEG -> {
                updateType(outs[0], args[0].type)
            }

            Prims.FLOOR,
            Prims.CEIL,
            Prims.ROUND,
            Prims.ASIN,
            Prims.SQRT,
            Prims.SIN -> {
                val at = args[0].type
                val ot = at.cast<ArrayType>()
                    ?.let { it.mapInner { Types.double } }
                    ?: at
                updateType(outs[0], ot)
            }

            Prims.RAND,
            Prims.REPLACE_RAND,
            Prims.NOW -> {
                updateType(outs[0], Types.double)
            }

            Prims.EQ,
            Prims.LT,
            Prims.ADD,
            Prims.SUB,
            Prims.MUL,
            Prims.MOD,
            Prims.MAX,
            Prims.MIN -> {
                val ty: Type = (args.mapNotNull { it.type as? ArrayType }.reduceOrNull { acc: ArrayType, irVar: ArrayType ->
                    List(max(irVar.shape.size, acc.shape.size)){-1}
                        .shapeToArrType(irVar.inner.combine(acc.inner))
                } ?: args.map { it.type }.reduce { a, b ->
                    a.combine(b)
                }).let {
                    // we don't want inaccuracy
                    if (it is AutoByteType) Types.int
                    else it
                }
                updateType(outs[0], ty)
            }

            Prims.PRIMES -> {
                updateType(outs[0], Types.array(Types.int))
            }

            Prims.RANGE -> {
                val at = args[0].type
                val ot = when (at) {
                    is ArrayType -> List(at.shape.size+1){-1}.shapeToArrType(at.inner)
                    else -> Types.array(at)
                }
                updateType(outs[0], ot)
            }

            Prims.DUP -> {
                val ty = args[0].type
                updateType(outs[0], ty)
                updateType(outs[1], ty)
            }

            Prims.FLIP -> {
                updateType(outs[0], args[1].type)
                updateType(outs[1], args[0].type)
            }

            Prims.OVER -> {
                updateType(outs[0], args[1].type)
                updateType(outs[1], args[0].type)
                updateType(outs[2], args[1].type)
            }

            Prims.LEN -> {
                updateType(outs[0], Types.size)
            }

            Prims.SWITCH -> {
                val at = parent.instrDeclFor(args[0])!!
                require(at.instr is PrimitiveInstr && at.instr.id == Prims.Comp.ARG_ARR)

                val fnsa = parent.instrDeclFor(args[1])!!
                require(fnsa.instr is PrimitiveInstr && fnsa.instr.id == Prims.Comp.ARG_ARR)
                val fns = fnsa.args

                val on = args[2]
                require(on.type is NumericType)

                val inputs = args.drop(3)

                if (verify) {
                    fns.forEach {
                        a.function(it)!!.type().let {
                            require(it.args.tyEq(inputs.map(IrVar::type)))
                            require(it.rets.tyEq(outs.map(IrVar::type)))
                            require(it.fillType == fillType)
                        }
                    }
                } else {
                    val expanded = fns.mapIndexed { index, arg ->
                        val fn = a.function(arg)!!
                        val exp = fn.expandFor(
                            inputs.map { it.type },
                            putFn,
                            fillType
                        )
                        val expb = parent.ref[exp]!!
                        fns[index] = fnRef(exp)
                        expb
                    }

                    require(expanded.all { it.rets.size == outs.size })

                    outs.zip(expanded.first().rets).forEach { (v, t) ->
                        updateType(v, t.type)
                    }
                }
            }

            Prims.REDUCE, Prims.Front.REDUCE_DEPTH -> {
                val fnblock = a.function(args[0])!!
                val inp = args[1].type as ArrayType

                val depth = if (instr.id == Prims.Front.REDUCE_DEPTH) instr.param!! else 0
                val inpOf = inp.into(depth + 1)

                if (verify) {

                } else {
                    val first = fnblock.expandFor(listOf(inpOf, inpOf), putFn, fillType)
                    val firstb = parent.ref[first]!!
                    val accTy = firstb.rets[0].type

                    args[0] = fnRef(first)

                    val all = fnblock.expandFor(listOf(accTy, inpOf), putFn, fillType)
                    val allb = parent.ref[all]!!
                    require(allb.rets[0].type == accTy)

                    args.add(fnRef(all))

                    updateType(outs[0], List(depth) { -1 }.shapeToType(accTy))
                }
            }

            Prims.EACH -> {
                val fnblock = a.function(args[0])!!

                val highestRank = args.drop(1)
                    .filter { it.type is ArrayType && it.type.length != 1 }
                    .maxBy { (it.type as ArrayType).shape.size }
                    .type as ArrayType

                val inps = args.drop(1).map {
                    if (it.type !is ArrayType || it.type.length == 1) it.type
                    else it.type.inner
                }

                val newb = if (verify) {
                    a.function(args[0])!!
                } else {
                    val new = fnblock.expandFor(inps, putFn, fillType)
                    args[0] = fnRef(new)
                    parent.ref[new]!!
                }

                outs.zip(newb.rets).forEach { (out, ret) ->
                    updateType(out, highestRank.mapInner { ret.type })
                }
            }

            Prims.SHAPE -> {
                val arg = args[0].type as ArrayType
                updateType(outs[0], Types.array(Types.size, arg.shape.size))
            }

            Prims.ROWS -> {
                val inps = args.drop(1).map {
                    it.type
                        .ofIfArray()
                        .makeVaOffIfArray()
                }

                val newb = if (verify) {
                    a.function(args[0])!!.also {
                        require(it.args.map { it.type }.tyEq(inps))
                    }
                } else {
                    val new = expandArgFn(0, inps, fillType)
                    args[0] = fnRef(new.uid)
                    new
                }

                outs.zip(newb.rets).forEach { (out, ret) ->
                    updateType(out, Types.array(ret.type))
                }
            }

            Prims.WHERE -> {
                val arrTy = args[0].type as ArrayType
                val shaLen = arrTy.shape.size

                val ty = if (shaLen == 1) Types.array(Types.size)
                else Types.array(Types.array(Types.size, length = shaLen))

                updateType(outs[0], ty)
            }

            Prims.FILL -> {
                val fillValFn = a.function(args[0])!!
                val opFn = a.function(args[1])!!
                val opArgs = args.drop(2)

                if (verify) {

                } else {
                    val fillValFnExp = fillValFn.expandFor(listOf(), putFn, fillType)
                    val fillValFnExpBlock = parent.ref[fillValFnExp]!!
                    val opFnExp = opFn.expandFor(opArgs.map { it.type }, putFn, fillValFnExpBlock.rets[0].type)

                    args[0] = fnRef(fillValFnExp)
                    args[1] = fnRef(opFnExp)

                    outs.zip(parent.ref[opFnExp]!!.rets).forEach { (out, ret) ->
                        updateType(out, ret.type)
                    }
                }
            }

            Prims.FIX,
            Prims.Comp.FIX_ARR -> {
                val argTy = args[0].type
                updateType(outs[0], Types.array(argTy, 1))
            }

            Prims.REVERSE -> {
                val argTy = args[0].type
                if (argTy is BoxType) {
                    require(argTy.of is ArrayType)
                } else {
                    require(argTy is ArrayType)
                }
                updateType(outs[0], argTy)
            }

            Prims.PICK -> {
                val arr = args[1].type as ArrayType
                updateType(outs[0], arr.of.makeVaOffIfArray())
            }

            Prims.UNDO_PICK -> {
                val arr = args[0].type as ArrayType
                updateType(outs[0], arr)
            }

            Prims.TABLE -> {
                val fn = a.function(args[0])!!
                val arr0 = args[1].type as ArrayType
                val arr1 = args[2].type as ArrayType

                val fnExpBlock = if (verify) {
                    fn
                } else {
                    expandArgFn(0, listOf(arr0.of, arr1.of), fillType)
                }

                outs.zip(fnExpBlock.rets).forEach { (out, ret) ->
                    updateType(out, Types.array(Types.array(ret.type)))
                }
            }

            Prims.RESHAPE -> {
                val sha = args[0]

                if (sha.type is ArrayType) {
                    val arr = args[1]
                    val innerTy = if (arr.type is ArrayType) arr.type.inner else arr.type

                    val shaLen = a
                        .deepOriginV2(sha)!!
                        .a!!.second
                        .instr.cast<ArrImmInstr>()!!
                        .values.flatten().size

                    val resTy = List(shaLen) { -1 }.shapeToArrType(innerTy)
                    updateType(outs[0], resTy)
                }
                else { // copy the value as rows of new array
                    val inner = args[1].type
                    val resTy = Types.array(inner)
                    updateType(outs[0], resTy)
                }
            }

            Prims.UN_SHAPE -> {
                val sha = args[0]
                val shaTy = sha.type as ArrayType

                require(shaTy.length != null) {
                    "length of shape has to be known at comptime!"
                }

                val resTy = List(shaTy.length) { -1 }.shapeToArrType(Types.int)

                updateType(outs[0], resTy)
            }

            Prims.JOIN -> {
                val arg0 = args[0].type
                val arg1 = args[1].type

                val highest = highestShapeType(arg0, arg1)
                val res = if (highest is ArrayType)
                    highest.mapShapeElems { -1 }
                else Types.array(highest, 2)
                updateType(outs[0], res)
            }

            Prims.KEEP -> {
                updateType(outs[0], args[1].type)
            }

            Prims.COMPLEX -> {
                updateType(outs[0], Types.complex)
            }

            Prims.RERANK -> {
                val arr = args[0]
                val rank = args[1]
                val ra = a.constNum(rank)
                    ?: error("rank to rerank needs to be immediate number")
                val inner = arr.type.innerIfArray()
                val res = List(ra.toInt() + 1) { -1 }.shapeToArrType(inner)
                updateType(outs[0], res)
            }

            Prims.UN_COUPLE -> {
                val arr = args[0].type as ArrayType
                updateType(outs[0], arr.of)
                updateType(outs[1], arr.of)
            }

            Prims.TRANSPOSE,
            Prims.Front.UN_TRANSPOSE -> {
                val arr = args[0].type as ArrayType
                updateType(outs[0], arr.mapShapeElems { -1 })
            }

            Prims.Comp.ARR_CLONE -> {
                updateType(outs[0], args[0].type)
            }

            Prims.Comp.ARR_COPY -> {
                require(args.map { (it.type as ArrayType).shape.size }.let { (a, b) -> a == b })
            }

            Prims.Comp.DYN_TYPEID -> {
                require(args[0].type == Types.dynamic)
                updateType(outs[0], Types.size)
            }

            Prims.Comp.DYN_UNWRAP -> {
                require(args[0].type == Types.dynamic)
            }

            Prims.Comp.DYN_WRAP -> {
                updateType(outs[0], Types.dynamic)
            }

            Prims.Comp.DYN_FREE -> {
                require(args[0].type == Types.dynamic)
            }

            Prims.Comp.TRANSPOSE -> {
                val arr = args[1].type as ArrayType
                updateType(args[0], arr.mapShapeElems { -1 })
            }

            Prims.Comp.PANIC,
            Prims.Comp.UNDEF,
            Prims.Comp.SINK -> {}

            Prims.Comp.RT_EXTEND_SCALAR,
            Prims.Comp.RT_EXTEND_REPEAT -> {}

            Prims.Comp.RESHAPE_VIEW -> {}

            else -> error("infer type for ${Prims.all[instr.id]} not implemented")
        }

        is SourceLocInstr -> {
            require(outs.size == 0)
        }
    }

    require(outs.none { it.type == Types.tbd }) {
        "type not inferred in op: $this"
    }
}
