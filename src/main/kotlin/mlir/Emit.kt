package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.ir.transform.constants
import kotlin.math.max

fun IrVar.asMLIR(): MLIRVar =
    "%${id}"

fun IrBlock.emitMLIR(): List<String> {
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

    fun castLaterIfNec(body: MutableList<String>, variable: IrVar, want: Type, block: (IrVar) -> Unit) {
        if (want is ArrayType && variable.type !is ArrayType)
            return castLaterIfNec(body, variable, want.inner, block)
        val dest = if (variable.type == want) variable
            else newVar().copy(type = want)
        block(dest)
        if (variable.type != want) {
            body.addAll(castInstr(
                ::newVar,
                from = want,
                to = variable.type,
                dest = variable.asMLIR(),
                src = dest.asMLIR()
            ))
        }
    }

    fun IrInstr.binary(
        body: MutableList<String>,
        op: (dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) -> String,
        reverse: Boolean = false
    ) {
        val rargs = if (reverse) args.reversed() else args
        val outTy = outs[0].type
        body += op(
            outs[0].asMLIR(),
            outTy.toMLIR(),
            castIfNec(body, rargs[0], outTy).asMLIR(),
            castIfNec(body, rargs[1], outTy).asMLIR(),
            outTy == Types.double
        )
    }

    fun callWithOptFill(dests: List<IrVar>, fn: IrBlock, args: List<IrVar>, fill: IrVar? = null): List<String> {
        if (fn.shouldInline()) {
            val toInline = fn.inlinableCopy(args, dests, fill)
            val c = toInline.emitMLIR()
            return c
        }

        return listOf(if (fn.fillArg != null) {
            Inst.funcCall(
                dests.map { it.asMLIR() },
                fn.name,
                fn.type().toMLIR(),
                *(listOf(fill!!) + args).map { it.asMLIR() }.toTypedArray()
            )
        } else {
            Inst.funcCall(
                dests.map { it.asMLIR() },
                fn.name,
                fn.type().toMLIR(),
                *args.map { it.asMLIR() }.toTypedArray()
            )
        })
    }

    fun callWithOptFill(dests: List<IrVar>, fn: IrVar, args: List<IrVar>, fill: IrVar? = null): List<String> {
        funDeclFor(fn)?.let { (_, block) ->
            return callWithOptFill(dests, block, args, fill)
        }

        val ty = fn.type as FnType
        return listOf(if (ty.fillType != null) {
            Inst.funcCallIndirect(
                dests.map { it.asMLIR() },
                fn.asMLIR(),
                ty.toMLIR(),
                (listOf(fill!!) + args).map { it.asMLIR() }
            )
        } else {
            Inst.funcCallIndirect(
                dests.map { it.asMLIR() },
                fn.asMLIR(),
                ty.toMLIR(),
                args.map { it.asMLIR() }.toList()
            )
        })
    }

    fun argArr(argArray: IrVar): List<IrVar> =
        instrDeclFor(argArray)!!.args

    fun subview(body: MutableList<String>, dest: IrVar, arr: IrVar, indecies: List<IrVar>) {
        val arrTy = arr.type as ArrayType
        val dims = List(arrTy.shape.size) { i ->
            val const = newVar().copy(type = Types.size)
            body += Inst.constant(const.asMLIR(), const.type.toMLIR(), "1")
            val v = newVar().copy(type = Types.size)
            body += Inst.memRefDim(v.asMLIR(), arrTy.toMLIR(), arr.asMLIR(), const.asMLIR())
            v
        }
        /*
        val offsets = indecies.map { it.asMLIR() } + List(arrTy.shape.size) { "0" }
        val size = arrTy.shape.mapIndexed { i, s -> if (i < indecies.size) 1 else s }.shapeToMLIR()
        val strides = arrTy.shape.map { "1" }
         */
        val offsets = indecies.map { castIfNec(body, it, Types.size).asMLIR() } + List(arrTy.shape.size - indecies.size) { 0 }
        val size = indecies.map { 1 }.shapeToMLIR() + List(arrTy.shape.size - indecies.size) { dims[it + indecies.size].asMLIR() }
        val strides = dims.map { it.asMLIR() }.drop(1).plus("1")

        body += "${dest.asMLIR()} = memref.subview ${arr.asMLIR()}[${offsets.joinToString()}][${size.joinToString()}][${strides.joinToString()}] : \n  ${arr.type.toMLIR()} to ${dest.type.toMLIR()}"
    }

    fun subview(body: MutableList<String>, arr: IrVar, indecies: List<IrVar>): IrVar {
        val arrTy = arr.type as ArrayType
        val dest = newVar().copy(type = arrTy.shape.drop(indecies.size).shapeToType(arrTy.inner).copy(vaOff = true))
        subview(body, dest, arr, indecies)
        return dest
    }

    fun cmp(instr: IrInstr, s: String, u: String) {
        val out = instr.outs[0]
        val outTy = out.type
        val a = castIfNec(body, instr.args[1], outTy).asMLIR()
        val b = castIfNec(body, instr.args[0], outTy).asMLIR()

        castLaterIfNec(body, out, Types.bool) { dest ->
            body += when (outTy) {
                Types.double -> "${dest.asMLIR()} = arith.cmpf $s, $a, $b : f64"
                Types.int -> "${dest.asMLIR()} = arith.cmpi $s, $a, $b : i64"
                Types.byte -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : i8"
                Types.size -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : index"
                else -> error("")
            }
        }
    }

    instrs.forEach { instr ->
        when (instr.instr) {
            is NumImmInstr -> {
                val value = instr.instr.value
                val ty = instr.outs[0].type
                val valueStr = when (ty) {
                    Types.int,
                    Types.byte,
                    Types.size,
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
                Prim.ADD -> instr.binary(body, Inst::add)
                Prim.SUB -> instr.binary(body, Inst::sub, reverse = true)
                Prim.MUL -> instr.binary(body, Inst::mul)
                Prim.DIV -> instr.binary(body, Inst::div, reverse = true)
                Prim.POW -> instr.binary(body, Inst::pow, reverse = true)

                Prim.LT -> {
                    cmp(instr, "slt", "ult")
                }
                Prim.EQ -> {
                    cmp(instr, "eq", "eq")
                }

                Prim.MAX -> {
                    val out = instr.outs[0]
                    val outTy = out.type
                    val a = castIfNec(body, instr.args[0], outTy).asMLIR()
                    val b = castIfNec(body, instr.args[1], outTy).asMLIR()

                    body += when (outTy) {
                        Types.double -> "${out.asMLIR()} = arith.maxnumf $a, $b : f64"
                        Types.int -> "${out.asMLIR()} = arith.maxsi $a, $b : i64"
                        Types.byte -> "${out.asMLIR()} = arith.maxui $a, $b : i8"
                        Types.size -> "${out.asMLIR()} = arith.maxui $a, $b : index"
                        else -> error("")
                    }
                }

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
                                dests,
                                target,
                                args,
                                fillArg
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

                    val temp = newVar().copy(type = NewlyAllocArrayType.from(type))

                    body += Inst.memRefAlloc(
                        temp.asMLIR(),
                        temp.type.toMLIR(),
                        *mShape.filterIndexed { i, _ -> shape[i] == -1 }.toTypedArray()
                    )

                    subview(body, instr.outs[0], temp, listOf())
                }

                Prim.Comp.ARR_STORE -> {
                    val arr = instr.args[0]
                    val arrTy = arr.type as ArrayType
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
                            castIfNec(body, value, arrTy.inner).asMLIR(),
                            arr.asMLIR(),
                            *indecies.map { castIfNec(body, it, Types.size).asMLIR() }.toTypedArray()
                        )
                    }
                }

                Prim.Comp.ARR_COPY -> {
                    body += Inst.memRefCopy(
                        instr.args[1].asMLIR(),
                        instr.args[1].type.toMLIR(),
                        instr.args[0].asMLIR(),
                        instr.args[0].type.toMLIR()
                    )
                }

                Prim.Comp.ARR_LOAD -> {
                    val arr = instr.args[0]
                    val indecies = argArr(instr.args[1])

                    if (instr.outs[0].type is ArrayType) {
                        // TODO: create subview primitive and use arr copy + arr load at higher level
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

                    val additional = instr.args.drop(3).toMutableList()

                    val counter = newVar().copy(type = Types.size)

                    val inner = mutableListOf<String>()

                    inner += callWithOptFill(
                        listOf(),
                        fn,
                        additional.also { it.add(0, castIfNec(inner, counter, fnTy.args[0])) },
                        fillArg
                    )

                    body += Inst.affineFor(
                        counter.asMLIR(),
                        start,
                        ends,
                        inner
                    )
                }

                Prim.Comp.DIM -> {
                    castLaterIfNec(body, instr.outs[0], Types.size) { dest ->
                        val dim = castIfNec(body, instr.args[1], Types.size)
                        body += Inst.memRefDim(
                            dest = dest.asMLIR(),
                            memRefType = instr.args[0].type.toMLIR(),
                            memRef = instr.args[0].asMLIR(),
                            dim = dim.asMLIR()
                        )
                    }
                }

                Prim.Comp.PANIC -> {
                    val rtPanic = Types.func(listOf(Types.int, Types.int), listOf())

                    val idBlock = newVar().asMLIR()
                    body += Inst.constant(idBlock, "i64", uid.toString())

                    val idInst = newVar().asMLIR()
                    body += Inst.constant(idInst, "i64", instrs.indexOf(instr).toString())

                    body += Inst.funcCall(
                        dests = listOf(),
                        "_\$_rt_panic",
                        rtPanic.toMLIR(),
                        idBlock, idInst
                    )

                    instr.outs.forEach {
                        body += Inst.undef(it.asMLIR(), it.type.toMLIR())
                    }
                }

                Prim.FILL -> {
                    val (_, fillValFn) = funDeclFor(instr.args[0])!!
                    val opFn = instr.args[1]
                    val opArgs = instr.args.drop(2)

                    val fillVal = newVar().copy(type = fillValFn.type().rets.first())
                    body += callWithOptFill(
                        listOf(fillVal),
                        fillValFn,
                        listOf(),
                    )

                    body += callWithOptFill(
                        instr.outs,
                        opFn,
                        opArgs,
                        fillVal
                    )
                }

                Prim.CALL -> {
                    val fn = instr.args[0]
                    val args = instr.args.drop(1)

                    body += callWithOptFill(
                        instr.outs,
                        fn,
                        args,
                        fillArg
                    )
                }

                else -> error("primitive ${instr.instr.id} not implemented")
            }

            is CommentInstr -> {}

            else -> error("$instr not implemented")
        }
    }

    return body
}

fun IrBlock.emitMLIRFinalize(body: List<String>): String {
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