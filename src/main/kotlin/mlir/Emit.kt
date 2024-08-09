package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.mlir.Inst.pDests

fun IrVar.asMLIR(): MLIRVar =
    "%${id}"

private val knowFns = mutableMapOf<IrVar, IrBlock>()

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
        (funDeclFor(fn)?.second ?: knowFns[fn])?.let { block ->
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
                Types.bool -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : i1"
                Types.size -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : index"
                else -> error("cmp not implemented for $outTy")
            }
        }
    }

    fun memRefCopy(dest: IrVar, src: IrVar) {
        if ((src.type as ArrayType).shape.size == 1) {
            val count = newVar().copy(type = Types.size).asMLIR()
            val zero = newVar().copy(type = Types.size).asMLIR()
            body += Inst.constant(zero, Types.size.toMLIR(), "0")
            val one = newVar().copy(type = Types.size).asMLIR()
            body += Inst.constant(one, Types.size.toMLIR(), "1")
            body += Inst.memRefDim(count, src.type.toMLIR(), src.asMLIR(), zero)
            val idx = newVar().copy(type = Types.size).asMLIR()

            val inner = mutableListOf<String>()

            val value = newVar().copy(type = src.type.of)
            inner += Inst.memRefLoad(value.asMLIR(), src.type.toMLIR(), src.asMLIR(), idx)
            inner += Inst.memRefStore(src.type.toMLIR(), value.asMLIR(), dest.asMLIR(), idx)

            Inst.scfFor(idx, zero, count, one, inner)
        } else {
            body += Inst.memRefCopy(
                src.asMLIR(),
                src.type.toMLIR(),
                dest.asMLIR(),
                dest.type.toMLIR()
            )
        }
    }

    instrs.forEachIndexed { idx, instr ->
        runCatching {
            when (instr.instr) {
                is NumImmInstr -> {
                    val value = instr.instr.value
                    val ty = instr.outs[0].type
                    val valueStr = when (ty) {
                        Types.int,
                        Types.byte,
                        Types.size,
                        Types.bool,
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
                    knowFns[instr.outs[0]] = ref[instr.instr.fn]!!
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
                        val conds = argArr(instr.args[0]).mapTo(mutableListOf()) {
                            (instrDeclFor(it)!!.instr as NumImmInstr).value.toULong()
                        }
                        val targets = argArr(instr.args[1]).toMutableList()
                        val on = castIfNec(body, instr.args[2], Types.size)
                        val args = instr.args.drop(3)

                        val terminating = conds.zip(targets).filter { (_, it) ->
                            funDeclFor(it)?.second?.terminating() == true
                        }

                        // we do this for now because terminating
                        terminating.forEach { (c, t) ->
                            conds.remove(c)
                            targets.remove(t)

                            val const = newVar().copy(type = Types.size)
                            body += "${const.asMLIR()} = arith.constant $c : index"

                            val cond = newVar().copy(type = Types.bool)
                            val inst = IrInstr(
                                mutableListOf(cond),
                                PrimitiveInstr(""),
                                mutableListOf(const, on)
                            )
                            cmp(inst, "eq", "eq")

                            val inner = mutableListOf<String>()
                            val dests = instr.outs.map { newVar().copy(type = it.type) }

                            inner += callWithOptFill(
                                dests,
                                t,
                                args,
                                fillArg
                            )

                            body += "scf.if ${cond.asMLIR()} {\n  ${inner.joinToString("\n  ")}\n}"
                        }

                        when (conds.size) {
                            0 -> {}
                            1 -> {
                                body += callWithOptFill(
                                    instr.outs,
                                    targets[0],
                                    args,
                                    fillArg
                                )
                            }
                            2 -> {
                                val const = newVar().copy(type = Types.size)
                                body += "${const.asMLIR()} = arith.constant ${conds[0]} : index"

                                val cond = newVar().copy(type = Types.bool)
                                val inst = IrInstr(
                                    mutableListOf(cond),
                                    PrimitiveInstr(""),
                                    mutableListOf(const, on)
                                )
                                cmp(inst, "eq", "eq")

                                fun gyield(dsts: List<IrVar>) =
                                    if (dsts.isEmpty()) "scf.yield"
                                    else "scf.yield ${dsts.map { it.asMLIR() }.joinToString()} : ${dsts.map { it.type.toMLIR() }.joinToString()}"

                                val then = mutableListOf<String>()
                                val thenDests = instr.outs.map { newVar().copy(type = it.type) }

                                then += callWithOptFill(
                                    thenDests,
                                    targets[0],
                                    args,
                                    fillArg
                                )

                                then += gyield(thenDests)

                                val els = mutableListOf<String>()
                                val elsDests = instr.outs.map { newVar().copy(type = it.type) }

                                els += callWithOptFill(
                                    elsDests,
                                    targets[1],
                                    args,
                                    fillArg
                                )

                                els += gyield(elsDests)

                                body += "${pDests(instr.outs.map { it.asMLIR() })}scf.if ${cond.asMLIR()} -> (${instr.outs.map { it.type.toMLIR() }.joinToString()}) {\n  ${then.joinToString("\n  ")}\n} else {\n  ${els.joinToString("\n  ")}\n}"
                            }
                            else -> {
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

                                        inner += "scf.yield ${dests.joinToString { it.asMLIR() }} ${if (instr.outs.isEmpty()) "" else ":"} ${dests.joinToString { it.type.toMLIR() }}"

                                        "$cond{\n  ${inner.joinToString("\n  ")}\n}"
                                    }
                                body += "${Inst.pDests(instr.outs.map { it.asMLIR() })}scf.index_switch ${on.asMLIR()} ${if (instr.outs.isEmpty()) "" else "->"} ${instr.outs.joinToString { it.type.toMLIR() }}\n${
                                    cases.joinToString(
                                        "\n"
                                    )
                                }"
                            }
                        }
                    }

                    Prim.Comp.ARG_ARR -> {} // ignore

                    Prim.Comp.ARR_MATERIALIZE -> TODO("in higher level impl arr materialize using alloc and rep store")

                    Prim.Comp.ARR_ALLOC -> {
                        val type = instr.outs[0].type as ArrayType
                        val shape = type.shape
                        val shapeDecl = instrDeclFor(instr.args[0])!!
                        val mShape = shapeDecl.args.map { castIfNec(body, it, Types.size).asMLIR() }

                        val temp = newVar().copy(type = NewlyAllocArrayType.from(type))

                        body += Inst.memRefAlloc(
                            temp.asMLIR(),
                            temp.type.toMLIR(),
                            *mShape.filterIndexed { i, _ -> shape[i] == -1 }.toTypedArray()
                        )

                        val temp2 = newVar().copy(type = type.copyVarShape().copyType())
                        subview(body, temp2, temp, listOf())
                        body += "${instr.outs[0].asMLIR()} = memref.cast ${temp2.asMLIR()} : ${temp2.type.toMLIR()} to ${type.toMLIR()}"
                    }

                    Prim.Comp.ARR_STORE -> {
                        val arr = instr.args[0]
                        val arrTy = arr.type as ArrayType
                        val indecies = argArr(instr.args[1])
                        val value = instr.args[2]

                        if (value.type is ArrayType) {
                            val view = subview(body, arr, indecies)
                            memRefCopy(view, value)
                        } else {
                            body += Inst.memRefStore(
                                arr.type.toMLIR(),
                                castIfNec(body, value, arrTy.inner).asMLIR(),
                                arr.asMLIR(),
                                *indecies.map { castIfNec(body, it, Types.size).asMLIR() }.toTypedArray()
                            )
                        }
                    }

                    Prim.Comp.ARR_COPY -> {
                        memRefCopy(instr.args[0], instr.args[1])
                    }

                    Prim.Comp.ARR_LOAD -> {
                        val arr = instr.args[0]
                        val indecies = argArr(instr.args[1])

                        if (instr.outs[0].type is ArrayType) {
                            // TODO: create subview primitive and use arr copy + arr load at higher level
                            subview(body, instr.outs[0], arr, indecies)
                        } else {
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

                        val one = newVar().copy(type = Types.size).asMLIR()
                        body += "$one = arith.constant 1 : index"

                        body += Inst.scfFor(
                            counter.asMLIR(),
                            start,
                            ends,
                            one,
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
                        val idBlock = newVar().asMLIR()
                        body += Inst.constant(idBlock, "i64", uid.toString())

                        val idInst = newVar().asMLIR()
                        body += Inst.constant(idInst, "i64", instrs.indexOf(instr).toString())

                        body += Inst.funcCall(
                            dests = listOf(),
                            UARuntime.panic.name,
                            UARuntime.panic.type.toMLIR(),
                            idBlock, idInst
                        )

                        instr.outs.forEach {
                            body += "${it.asMLIR()} = ub.poison : ${it.type.toMLIR()}"
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

                    Prim.RESHAPE -> {
                        // TODO: uiua reshape vs compiler reshape:  uiua reshape lowers to clone and then compieler reshape

                        val sha = argArr(instr.args[0])
                            .map { castIfNec(body, it, Types.size) }

                        val arr = instr.args[1]
                        val arrTy = arr.type as ArrayType

                        val out = instr.outs[0]

                        body += "${out.asMLIR()} = memref.view ${arr.asMLIR()}[][${sha.joinToString()}] :\n  ${arrTy.toMLIR()} to ${out.type.toMLIR()}"
                    }

                    else -> error("primitive ${instr.instr.id} not implemented")
                }

                is CommentInstr -> {}

                else -> error("$instr not implemented")
            }
        }.onFailure {
            println("in mlir emit instr $uid:$idx: ${it.message}")
            throw it
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