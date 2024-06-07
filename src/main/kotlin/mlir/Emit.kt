package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun IrVar.asMLIR(): MLIRVar =
    "%${id}"

fun IrBlock.emitMLIR(): String {
    val body = mutableListOf<String>()

    fun castIfNec(body: MutableList<String>, variable: IrVar, want: Type): IrVar =
        if (want is ArrayType && variable.type !is ArrayType)
            castIfNec(body, variable, want.inner)
        else
            if (variable.type == want) variable
            else {
                val new = newVar().copy(type = want)
                body.addAll(castInstr(
                    ::newVar,
                    from = variable.type,
                    to = want,
                    dest = new.asMLIR(),
                    src = variable.asMLIR()
                ))
                new
            }

    fun emitShapeOf(body: MutableList<String>, src: IrVar): List<MLIRVar> {
        val srcTy = src.type as ArrayType

        return List(srcTy.shape.size) {
            val const = newVar().asMLIR()
            body += Inst.constant(const, Ty.index, it.toString())
            val v = newVar().asMLIR()
            body += Inst.memRefDim(v, srcTy.toMLIR(), src.asMLIR(), const)
            v
        }
    }

    fun IrInstr.binary(
        body: MutableList<String>,
        op: (dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) -> String,
        opArr: (dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) -> String,
        reverse: Boolean = false
    ) {
        val rargs = if (reverse) args.reversed() else args
        val outTy = outs[0].type
        if (outTy is ArrayType) {
            val arrArg = rargs.find { it.type is ArrayType }
            val cast = rargs.map {
                if (it.type is ArrayType) it
                else castIfNec(body, it, outTy)
            }

            val mDynShape = arrArg?.let {
                val mShape = emitShapeOf(body, it)
                val arrArgTy = it.type as ArrayType
                mShape.filterIndexed { idx, _ -> arrArgTy.shape[idx] == -1 }
            } ?: listOf()

            body += Inst.memRefAlloc(
                outs[0].asMLIR(),
                outTy.toMLIR(),
                *mDynShape.toTypedArray()
            )
            body += opArr(
                outs[0].asMLIR(),
                outs[0].type.toMLIR(),
                cast.map { it.asMLIR() to it.type.toMLIR() }
            )
        } else {
            body += op(
                outs[0].asMLIR(),
                outTy.toMLIR(),
                castIfNec(body, rargs[0], outTy).asMLIR(),
                castIfNec(body, rargs[1], outTy).asMLIR(),
                outTy == Types.double
            )
        }
    }

    fun callWithOptFill(dests: List<MLIRVar>, fn: IrBlock, vararg args: MLIRVar,  fill: IrVar? = null): String =
        if (fn.fillArg != null) {
            Inst.funcCall(
                dests,
                fn.name,
                fn.type().toMLIR(),
                *(arrayOf(fill!!.asMLIR()) + args)
            )
        } else {
            Inst.funcCall(
                dests,
                fn.name,
                fn.type().toMLIR(),
                *args
            )
        }

    fun callWithOptFill(dests: List<MLIRVar>, fn: IrVar, vararg args: MLIRVar, fill: IrVar? = null): String {
        val ty = fn.type as FnType
        return if (ty.fillType != null) {
            Inst.funcCallIndirect(
                dests,
                fn.asMLIR(),
                ty.toMLIR(),
                listOf(fill!!.asMLIR()) + args
            )
        } else {
            Inst.funcCallIndirect(
                dests,
                fn.asMLIR(),
                ty.toMLIR(),
                args.toList()
            )
        }
    }

    fun argArr(argArray: IrVar): List<IrVar> =
        instrDeclFor(argArray)!!.args

    fun subview(body: MutableList<String>, dest: IrVar, arr: IrVar, indecies: List<IrVar>) {
        val arrTy = arr.type as ArrayType

        val offsets = indecies.map { it.asMLIR() } + List(arrTy.shape.size) { "0" }
        val size = arrTy.shape.mapIndexed { i, s -> if (i < indecies.size) 1 else s }.shapeToMLIR()
        val strides = arrTy.shape.map { "1" }

        body += "${dest.asMLIR()} = memref.subview ${arr.asMLIR()}[${offsets.joinToString()}][${size.joinToString()}][${strides.joinToString()}] : \n  ${arr.type.toMLIR()} to ${dest.type.toMLIR()}"
    }

    fun subview(body: MutableList<String>, arr: IrVar, indecies: List<IrVar>): IrVar {
        val arrTy = arr.type as ArrayType
        val dest = newVar().copy(type = arrTy.shape.drop(indecies.size).shapeToType(arrTy.inner))
        subview(body, dest, arr, indecies)
        return dest
    }

    instrs.forEach { instr ->
        when (instr.instr) {
            is NumImmInstr -> {
                val value = instr.instr.value
                val ty = instr.outs[0].type
                val valueStr = when (ty) {
                    Types.int,
                    Types.byte,
                    is PtrType -> value.toULong().toString()
                    else -> value.toString()
                }
                body += Inst.constant(
                    instr.outs[0].asMLIR(),
                    ty.toMLIR(),
                    valueStr
                )
            }

            is PushFnRefInstr -> {
                val fn = instr.instr.fn.legalizeMLIR()
                body += Inst.funcConstant(
                    dest = instr.outs[0].asMLIR(),
                    fn = "@$fn",
                    fnType = instr.outs[0].type.toMLIR()
                )
            }

            is PrimitiveInstr -> when (instr.instr.id) {
                Prim.ADD -> instr.binary(body, Inst::add, Inst::arrAdd)
                Prim.SUB -> instr.binary(body, Inst::sub, Inst::arrSub)
                Prim.MUL -> instr.binary(body, Inst::mul, Inst::arrMul)
                Prim.DIV -> instr.binary(body, Inst::div, Inst::arrDiv, true)
                Prim.POW -> instr.binary(body, Inst::pow, Inst::arrPow, true)

                Prim.PRIMES -> {
                    val rtPrimes = Types.func(
                        args = listOf(Types.int),
                        rets = listOf(Types.array(Types.int))
                    )

                    body += Inst.funcCall(
                        dests = listOf(instr.outs[0].asMLIR()),
                        fn = "_\$_rt_primes",
                        fnType = rtPrimes.toMLIR(),
                        castIfNec(body, instr.args[0], Types.int).asMLIR()
                    )
                }

                Prim.SWITCH -> {
                    val conds = argArr(instr.args[0]).map {
                        (instrDeclFor(it)!!.instr as NumImmInstr).value.toULong()
                    }
                    val targets = argArr(instr.args[1])
                    val on = castIfNec(body, instr.args[2], Types.size)
                    val args = instr.args.drop(3)

                    val cases = (conds.dropLast(1).map { "case $it " } + conds.last().let { "default " })
                        .zip(targets)
                        .map { (cond, target) ->
                            val inner = mutableListOf<String>()
                            val dests = instr.outs.map { newVar().copy(type = it.type) }

                            inner += callWithOptFill(
                                dests = dests.map { it.asMLIR() },
                                fn = target,
                                *args.map { it.asMLIR() }.toTypedArray(),
                                fill = fillArg
                            )

                            inner += "scf.yield ${dests.joinToString { it.asMLIR() }} : ${dests.joinToString { it.type.toMLIR() }}"

                            "$cond{\n  ${inner.joinToString("\n  ")}\n}"
                        }
                    body += "${Inst.pDests(instr.outs.map { it.asMLIR() })}scf.index_switch ${on.asMLIR()} -> ${instr.outs.joinToString { it.type.toMLIR() }}\n${cases.joinToString("\n")}"
                }

                Prim.Comp.ARG_ARR -> {} // ignore

                Prim.Comp.ARR_MATERIALIZE -> TODO()

                Prim.Comp.ARR_ALLOC -> {
                    val type = instr.outs[0].type as ArrayType
                    val shape = type.shape
                    val mShape = argArr(instr.args[0]).map { castIfNec(body, it, Types.size).asMLIR() }

                    body += Inst.memRefAlloc(
                        instr.outs[0].asMLIR(),
                        type.toMLIR(),
                        *mShape.filterIndexed { i, _ -> shape[i] == -1 }.toTypedArray()
                    )
                }

                Prim.Comp.ARR_STORE -> {
                    val arr = instr.args[0]
                    val indecies = argArr(instr.args[1])
                    val value = instr.args[2]

                    if (value.type is ArrayType) {
                        val view = subview(body, arr, indecies)
                        body += Inst.memRefCopy(
                            value.asMLIR(),
                            value.type.toMLIR(),
                            view.asMLIR(),
                            view.type.toMLIR()
                        )
                    }
                    else {
                        body += Inst.memRefStore(
                            arr.type.toMLIR(),
                            value.asMLIR(),
                            arr.asMLIR(),
                            *indecies.map { castIfNec(body, it, Types.size).asMLIR() }.toTypedArray()
                        )
                    }
                }

                Prim.Comp.ARR_LOAD -> {
                    val arr = instr.args[0]
                    val indecies = argArr(instr.args[1])

                    if (instr.outs[0].type is ArrayType) {
                        subview(body, instr.outs[0], arr, indecies)
                    }
                    else {
                        body += Inst.memRefLoad(
                            instr.outs[0].asMLIR(),
                            arr.type.toMLIR(),
                            arr.asMLIR(),
                            *indecies.map { castIfNec(body, it, Types.size).asMLIR() }.toTypedArray()
                        )
                    }
                }

                Prim.Comp.ARR_DESTROY -> {
                    body += Inst.memRefDealloc(
                        instr.args[0].asMLIR(),
                        instr.args[0].type.toMLIR()
                    )
                }

                Prim.Comp.REPEAT -> {
                    val start = castIfNec(body, instr.args[0], Types.size).asMLIR()

                    val ends = castIfNec(body, instr.args[1], Types.size).asMLIR()
                    val fn = instr.args[2]
                    val fnTy = fn.type as FnType

                    val additional = instr.args.drop(3).mapTo(mutableListOf()) { it.asMLIR() }

                    val counter = newVar().copy(type = Types.size)

                    val inner = mutableListOf<String>()

                    inner += callWithOptFill(
                        listOf(),
                        fn,
                        *additional.also { it.add(0, castIfNec(inner, counter, fnTy.args[0]).asMLIR()) }.toTypedArray(),
                        fill = fillArg
                    )

                    body += Inst.affineParallelFor(
                        listOf(counter.asMLIR()),
                        listOf(start),
                        listOf(ends),
                        inner
                    )
                }

                Prim.Comp.DIM -> {
                    val dim = castIfNec(body, instr.args[1], Types.size)
                    body += Inst.memRefDim(
                        dest = instr.outs[0].asMLIR(),
                        memRefType = instr.args[0].type.toMLIR(),
                        memRef = instr.args[0].asMLIR(),
                        dim = dim.asMLIR()
                    )
                }

                Prim.Comp.PANIC -> {
                    val rtPanic = Types.func(listOf(), listOf())

                    body += Inst.funcCall(
                        dests = listOf(),
                        "_\$_rt_panic",
                        rtPanic.toMLIR()
                    )

                    instr.outs.forEach {
                        body += Inst.undef(it.asMLIR(), it.type.toMLIR())
                    }
                }

                Prim.FILL -> {
                    val (_, fillValFn) = funDeclFor(instr.args[0])
                    val opFn = instr.args[1]
                    val opArgs = instr.args.drop(2)

                    val fillVal = newVar()
                    body += callWithOptFill(
                        dests = listOf(fillVal.asMLIR()),
                        fn = fillValFn,
                    )

                    body += callWithOptFill(
                        dests = instr.outs.map { it.asMLIR() },
                        fn = opFn,
                        *opArgs.map { it.asMLIR() }.toTypedArray(),
                        fill = fillVal
                    )
                }

                Prim.CALL -> {
                    val fn = instr.args[0]
                    val args = instr.args.drop(1)

                    body += callWithOptFill(
                        instr.outs.map { it.asMLIR() },
                        fn = fn,
                        *args.map { it.asMLIR() }.toTypedArray(),
                        fill = fillArg
                    )
                }

                else -> error("")
            }

            else -> error("$instr not implemented")
        }
    }

    val mArgs = args.mapTo(mutableListOf()) { it.asMLIR() to it.type.toMLIR() }
    fillArg?.let {
        mArgs.add(0, it.asMLIR() to it.type.toMLIR())
    }

    return function(
        name,
        private,
        mArgs,
        rets.map { it.asMLIR() to it.type.toMLIR() },
        body
    )
}