package me.alex_s168.uiua.mlir

import blitz.collections.contents
import blitz.collections.mapToArray
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.mlir.Inst.pDests

fun IrVar.asMLIR(): MLIRVar =
    "%${id}"

private val knowFns = mutableMapOf<IrVar, IrBlock>()

fun IrBlock.emitMLIR(dbgInfoConsumer: (SourceLocInstr) -> List<String>): List<String> {
    val body = mutableListOf<String>()

    fun IrInstr.unary(
        body: MutableList<String>,
        op: (dest: MLIRVar, type: MLIRType, a: MLIRVar, float: Boolean) -> String
    ) {
        val outTy = outs[0].type
        body += op(
            outs[0].asMLIR(),
            outTy.toMLIR(),
            castIfNec(::newVar, body, args[0], outTy).asMLIR(),
            outTy == Types.double
        )
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
            castIfNec(::newVar, body, rargs[0], outTy).asMLIR(),
            castIfNec(::newVar, body, rargs[1], outTy).asMLIR(),
            outTy == Types.double
        )
    }

    fun genCall(dests: List<IrVar>, fn: IrBlock, args: List<IrVar>): List<String> {
        if (fn.shouldInline()) {
            val toInline = fn.inlinableCopy(args, dests)
            val res = mutableListOf<String>()
            if (mlirComments)
                res += "// Inlined ${fn.name} (${args.contents}) -> (${dests.contents})"
            res.addAll(toInline.emitMLIR(dbgInfoConsumer))
            return res
        }

        return listOf(Inst.funcCall(
            dests.map { it.asMLIR() },
            fn.name.legalizeMLIR(),
            fn.type().toMLIR(),
            *args.mapToArray { it.asMLIR() }
        ))
    }

    fun genCall(dests: List<IrVar>, fn: IrVar, argsIn: List<IrVar>): List<String> {
        val ty = fn.type as FnType

        require(ty.args.size == argsIn.size)
        val args = argsIn.zip(ty.args)
            .map { (it, want) -> castIfNec(::newVar, body, it, want) }

        (funDeclFor(fn) ?: knowFns[fn])?.let { block ->
            return genCall(dests, block, args)
        }

        return listOf(Inst.funcCallIndirect(
            dests.map { it.asMLIR() },
            fn.asMLIR(),
            ty.toMLIR(),
            args.map { it.asMLIR() }
        ))
    }

    val analysis = Analysis(this)
    val cache = CallerInstrsCache()

    fun argArr(argArray: IrVar): List<IrVar> =
        analysis.argArr(argArray, cache::get)?.b!!

    fun cmp(instr: IrInstr, s: String, u: String) {
        val out = instr.outs[0]
        val cmpTy = instr.args[0].type // TODO: find common between both
        val a = castIfNec(::newVar, body, instr.args[1], cmpTy).asMLIR()
        val b = castIfNec(::newVar, body, instr.args[0], cmpTy).asMLIR()

        castLaterIfNec(::newVar, body, out, Types.bool) { dest ->
            body += when (cmpTy) {
                Types.double -> "${dest.asMLIR()} = arith.cmpf $s, $a, $b : f64"
                Types.int -> "${dest.asMLIR()} = arith.cmpi $s, $a, $b : i64"
                Types.byte -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : i8"
                Types.bool -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : i1"
                Types.size -> "${dest.asMLIR()} = arith.cmpi $u, $a, $b : index"
                else -> error("cmp not implemented for $cmpTy")
            }
        }
    }

    fun memRefCopy(dest: IrVar, src: IrVar) {
        // TODO: in higher level, so that you can unroll
        if ((src.type as ArrayType).shape.size == 1) {
            if (mlirComments)
                body += "// memRefCopy ${src.asMLIR()} to ${dest.asMLIR()}"

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
            val real = castIfNec(::newVar, inner, value, (dest.type as ArrayType).inner)
            inner += Inst.memRefStore(dest.type.toMLIR(), real.asMLIR(), dest.asMLIR(), idx)

            body += Inst.scfFor(idx, zero, count, one, inner)
        } else {
            body += Inst.memRefCopy(
                src.asMLIR(),
                src.type.toMLIR(),
                dest.asMLIR(),
                dest.type.toMLIR()
            )
        }
    }

    repeat(instrs.size) { idx ->
        val instr = instrs[idx]

        if (mlirComments)
            body += "// Instr: $instr"
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
                    knowFns[instr.outs[0]] = ref[instr.instr.fn]
                        ?: error("sym ${instr.instr.fn} not found")
                    val fn = ref[instr.instr.fn]!!.name.legalizeMLIR()
                    body += Inst.funcConstant(
                        dest = instr.outs[0].asMLIR(),
                        fn = "@$fn",
                        fnType = instr.outs[0].type.toMLIR()
                    )
                }

                is SourceLocInstr -> {
                    body += dbgInfoConsumer(instr.instr)
                }

                is PrimitiveInstr -> when (instr.instr.id) {
                    Prims.ADD -> instr.binary(body, Inst::add)
                    Prims.SUB -> instr.binary(body, Inst::sub, reverse = true)
                    Prims.MUL -> instr.binary(body, Inst::mul)
                    Prims.DIV -> instr.binary(body, Inst::div, reverse = true)
                    Prims.MOD -> instr.binary(body, Inst::mod, reverse = true)
                    Prims.POW -> instr.binary(body, Inst::pow, reverse = true)

                    Prims.SQRT -> instr.unary(body, Inst::sqrt)
                    Prims.NEG -> instr.unary(body, Inst::neg)
                    Prims.SIN -> instr.unary(body, Inst::sin)
                    Prims.ASIN -> instr.unary(body, Inst::asin)
                    Prims.FLOOR -> instr.unary(body, Inst::floor)
                    Prims.CEIL -> instr.unary(body, Inst::ceil)
                    Prims.ROUND -> instr.unary(body, Inst::round)

                    Prims.LT -> {
                        cmp(instr, "slt", "ult")
                    }

                    Prims.EQ -> {
                        cmp(instr, "eq", "eq")
                    }

                    Prims.MAX,
                    Prims.MIN -> {
                        val out = instr.outs[0]
                        val outTy = out.type
                        val a = castIfNec(::newVar, body, instr.args[0], outTy).asMLIR()
                        val b = castIfNec(::newVar, body, instr.args[1], outTy).asMLIR()

                        val str = if (instr.instr.id == Prims.MAX) "max" else "min"
                        body += when (outTy) {
                            Types.double -> "${out.asMLIR()} = arith.${str}numf $a, $b : f64"
                            Types.int -> "${out.asMLIR()} = arith.${str}si $a, $b : i64"
                            Types.byte -> "${out.asMLIR()} = arith.${str}ui $a, $b : i8"
                            Types.size -> "${out.asMLIR()} = arith.${str}ui $a, $b : index"
                            else -> error("")
                        }
                    }

                    Prims.PRIMES -> {
                        val rtPrimes = Types.func(
                            args = listOf(Types.int),
                            rets = listOf(Types.array(Types.int))
                        )

                        // TODO: move to UARuntime
                        body += Inst.funcCall(
                            dests = listOf(instr.outs[0].asMLIR()),
                            fn = "_\$_rt_primes",
                            fnType = rtPrimes.toMLIR(),
                            castIfNec(::newVar, body, instr.args[0], Types.int).asMLIR()
                        )
                    }

                    Prims.Comp.DYN_TYPEID -> {
                        castLaterIfNec(::newVar, body, instr.outs[0], Types.byte) {
                            body += Ty.dyn.getVal(it.asMLIR(), instr.args[0].asMLIR(), 0)
                        }
                    }

                    Prims.Comp.DYN_UNWRAP -> {
                        val wantType = instr.instr.typeParam!!
                        val fn = UARuntime.dyn.castTo(wantType)
                        body += Inst.funcCall(
                            dests = listOf(instr.outs[0].asMLIR()),
                            fn = fn.name,
                            fnType = fn.type.toMLIR(),
                            instr.args[0].asMLIR()
                        )
                    }

                    Prims.Comp.DYN_WRAP -> {
                        val fn = UARuntime.dyn.createFrom(instr.args[0].type)
                        body += Inst.funcCall(
                            listOf(instr.outs[0].asMLIR()),
                            fn.name,
                            fn.type.toMLIR(),
                            instr.args[0].asMLIR(),
                        )
                    }

                    Prims.Comp.DYN_FREE -> {
                        val fn = UARuntime.dyn.drop
                        body += Inst.funcCall(
                            dests = listOf(),
                            fn = fn.name,
                            fnType = fn.type.toMLIR(),
                            instr.args[0].asMLIR(),
                        )
                    }

                    Prims.SWITCH -> {
                        val conds = argArr(instr.args[0]).mapTo(mutableListOf()) {
                            (instrDeclFor(it)!!.instr as NumImmInstr).value.toULong()
                        }
                        val targets = argArr(instr.args[1]).toMutableList()
                        val on = castIfNec(::newVar, body, instr.args[2], Types.size)
                        val args = instr.args.drop(3)

                        when (conds.size) {
                            0 -> {}
                            1 -> {
                                body += genCall(
                                    instr.outs,
                                    targets[0],
                                    args
                                )
                            }
                            2 -> {
                                val const = newVar().copy(type = Types.size)
                                body += "${const.asMLIR()} = arith.constant ${conds[0]} : index"

                                val cond = newVar().copy(type = Types.bool)
                                val inst = IrInstr(
                                    mutableListOf(cond),
                                    PrimitiveInstr(Prims.Comp.INVALID),
                                    mutableListOf(const, on)
                                )
                                cmp(inst, "eq", "eq")

                                fun gyield(dsts: List<IrVar>) =
                                    if (dsts.isEmpty()) "scf.yield"
                                    else "scf.yield ${dsts.map { it.asMLIR() }.joinToString()} : ${dsts.map { it.type.toMLIR() }.joinToString()}"

                                val then = mutableListOf<String>()
                                val thenDests = instr.outs.map { newVar().copy(type = it.type) }

                                then += genCall(
                                    thenDests,
                                    targets[0],
                                    args
                                )

                                then += gyield(thenDests)

                                val els = mutableListOf<String>()
                                val elsDests = instr.outs.map { newVar().copy(type = it.type) }

                                els += genCall(
                                    elsDests,
                                    targets[1],
                                    args
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

                                        inner += genCall(
                                            dests,
                                            target,
                                            args
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

                    Prims.Comp.ARG_ARR -> {}

                    Prims.Comp.ARR_ALLOC,
                    Prims.Comp.EMIT_ARR_ALLOC_P -> {
                        val (outVar, shapeVar) =
                            if (instr.instr.id == Prims.Comp.EMIT_ARR_ALLOC_P)
                                instr.args
                            else listOf(instr.outs[0], instr.args[0])

                        val type = outVar.type as ArrayType
                        val shape = type.shape
                        val shapeDecl = instrDeclFor(shapeVar)!!
                        val mShape = shapeDecl.args.map { castIfNec(::newVar, body, it, Types.size).asMLIR() }

                        val temp = newVar().copy(type = NewlyAllocArrayType.from(type))

                        body += Inst.memRefAlloc(
                            temp.asMLIR(),
                            temp.type.toMLIR(),
                            *mShape.filterIndexed { i, _ -> shape[i] == -1 }.toTypedArray()
                        )

                        body += "${outVar.asMLIR()} = memref.cast ${temp.asMLIR()} : ${temp.type.toMLIR()} to ${type.toMLIR()}"
                    }

                    Prims.Comp.ARR_STORE -> {
                        val arr = instr.args[0]
                        val arrTy = arr.type as ArrayType
                        val indecies = argArr(instr.args[1])
                        val value = instr.args[2]

                        if (value.type is ArrayType) {
                            // TODO: do in higher level
                            val view = subview(::newVar, body, arr, indecies)
                                .let { it.copy(type = (it.type as ArrayType).copy(vaOff = true)) }
                            memRefCopy(view, value)
                        } else {
                            body += Inst.memRefStore(
                                arr.type.toMLIR(),
                                castIfNec(::newVar, body, value, arrTy.inner).asMLIR(),
                                arr.asMLIR(),
                                *indecies.mapToArray { castIfNec(::newVar, body, it, Types.size).asMLIR() }
                            )
                        }
                    }

                    Prims.Comp.ARR_COPY -> error("no")

                    Prims.Comp.ARR_LOAD -> {
                        val arr = instr.args[0]
                        val indecies = argArr(instr.args[1])

                        if (instr.outs[0].type is ArrayType) {
                            // TODO: create subview primitive and use arr copy + arr load at higher level
                            subview(::newVar, body, instr.outs[0], arr, indecies)
                        } else {
                            castLaterIfNec(::newVar, body, instr.outs[0], arr.type) { dest ->
                                body += Inst.memRefLoad(
                                    dest.asMLIR(),
                                    arr.type.toMLIR(),
                                    arr.asMLIR(),
                                    *indecies.mapToArray { castIfNec(::newVar, body, it, Types.size).asMLIR() }
                                )
                            }
                        }
                    }

                    Prims.Comp.ARR_DESTROY -> {
                        body += Inst.memRefDealloc(
                            instr.args[0].asMLIR(),
                            instr.args[0].type.toMLIR()
                        )
                    }

                    Prims.Comp.REPEAT -> {
                        val start = castIfNec(::newVar, body, instr.args[0], Types.size).asMLIR()

                        val ends = castIfNec(::newVar, body, instr.args[1], Types.size).asMLIR()
                        val fn = instr.args[2]
                        val fnTy = fn.type as FnType

                        val additional = instr.args.drop(3).toMutableList()

                        val counter = newVar().copy(type = Types.size)

                        val inner = mutableListOf<String>()

                        inner += genCall(
                            listOf(),
                            fn,
                            additional.also {
                                it.add(0, castIfNec(::newVar, inner, counter, fnTy.args[0]))
                            }
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

                    Prims.Comp.DIM -> {
                        castLaterIfNec(::newVar, body, instr.outs[0], Types.size) { dest ->
                            val dim = castIfNec(::newVar, body, instr.args[1], Types.size)
                            body += Inst.memRefDim(
                                dest = dest.asMLIR(),
                                memRefType = instr.args[0].type.toMLIR(),
                                memRef = instr.args[0].asMLIR(),
                                dim = dim.asMLIR()
                            )
                        }
                    }

                    Prims.NOW -> {
                        body += Inst.funcCall(
                            dests = listOf(instr.outs[0].asMLIR()),
                            UARuntime.time.name,
                            UARuntime.time.type.toMLIR(),
                        )
                    }

                    Prims.Comp.RT_EXTEND_SCALAR -> {
                        val vspan = instr.instr.loc
                            ?.toString()
                            ?: "-1"

                        val span = newVar().asMLIR()
                        body += Inst.constant(span, "i64", vspan)

                        val fn = UARuntime.extendScalar(instr.args.map { it.type })
                        body += Inst.funcCall(
                            dests = instr.outs.map { it.asMLIR() },
                            fn.name,
                            fn.type.toMLIR(),
                            * (listOf(span) + instr.args.map { it.asMLIR() }).toTypedArray()
                        )
                    }

                    Prims.Comp.RT_EXTEND_REPEAT -> {
                        val vspan = instr.instr.loc
                            ?.toString()
                            ?: "-1"

                        val span = newVar().asMLIR()
                        body += Inst.constant(span, "i64", vspan)

                        val fn = UARuntime.extendRepeat(instr.args.map { it.type })
                        body += Inst.funcCall(
                            dests = instr.outs.map { it.asMLIR() },
                            fn.name,
                            fn.type.toMLIR(),
                            * (listOf(span) + instr.args.map { it.asMLIR() }).toTypedArray()
                        )
                    }

                    Prims.Comp.PANIC -> {
                        val idBlock = newVar().asMLIR()
                        body += Inst.constant(idBlock, "i64", uid.toString())

                        val idInst = newVar().asMLIR()
                        body += Inst.constant(idInst, "i64", instrs.indexOf(instr).toString())

                        val vspan = instr.instr.loc
                            ?.index
                            ?.toString()
                            ?: "-1"

                        val span = newVar().asMLIR()
                        body += Inst.constant(span, "i64", vspan)

                        body += Inst.funcCall(
                            dests = listOf(),
                            UARuntime.panic.name,
                            UARuntime.panic.type.toMLIR(),
                            span, idBlock, idInst
                        )

                        instr.outs.forEach {
                            body += "${it.asMLIR()} = ub.poison : ${it.type.toMLIR()}"
                        }
                    }

                    Prims.Comp.UNDEF -> {
                        instr.outs.forEach {
                            body += "${it.asMLIR()} = ub.poison : ${it.type.toMLIR()}"
                        }
                    }

                    Prims.Comp.SINK -> {
                        // do nothing
                    }

                    Prims.CALL -> {
                        val fn = instr.args[0]
                        val args = instr.args.drop(1)

                        body += genCall(
                            instr.outs,
                            fn,
                            args
                        )
                    }

                    Prims.Comp.FIX_ARR -> {
                        val arr = instr.args[0]
                        val arrTy = arr.type as ArrayType

                        val out = instr.outs[0]

                        val dims = instr.args.drop(1)
                        require(dims.size == arrTy.shape.size)

                        body += "${out.asMLIR()} = memref.expand_shape ${arr.asMLIR()} [[${List(arrTy.shape.size){it}.joinToString()}], [${arrTy.shape.size}]] output_shape [1, ${dims.map{it.asMLIR()}.joinToString()}] : ${arrTy.toMLIR()} into ${out.type.toMLIR()}"
                    }

                    Prims.Comp.RESHAPE_VIEW -> {
                        val sha = instr.args[0]
                        require((sha.type as ArrayType).length != null) {
                            "reshape shape array needs to have comptime known length"
                        }

                        val arr = instr.args[1]
                        val arrTy = arr.type as ArrayType

                        val out = instr.outs[0]
                        val outTy = out.type as ArrayType

                        val t0 = newVar().copy(type = NewlyAllocArrayType.from(arrTy))
                        body += "${t0.asMLIR()} = memref.cast ${arr.asMLIR()} : ${arrTy.toMLIR()} to ${t0.type.toMLIR()}"
                        val t1 = newVar().copy(type = NewlyAllocArrayType.from(outTy))
                        body += "${t1.asMLIR()} = memref.reshape ${t0.asMLIR()}(${sha.asMLIR()}) :\n  (${t0.type.toMLIR()}, ${sha.type.toMLIR()}) -> ${t1.type.toMLIR()}"
                        body += "${out.asMLIR()} = memref.cast ${t1.asMLIR()} : ${t1.type.toMLIR()} to ${outTy.toMLIR()}"
                    }

                    Prims.Comp.OFF_VIEW_1D -> { // [arr], [begin idx], [len]
                        val arr = instr.args[0]
                        val arrTy = arr.type as ArrayType
                        val off = instr.args[1]
                        val len = instr.args[2]

                        val out = instr.outs[0]
                        TODO("not implemented yet")
                        // need to use subview instead!!!!
                        //body += "${out.asMLIR()} = memref.view ${arr.asMLIR()}[${off.asMLIR()}][${len.asMLIR()}] :\n  ${arrTy.toMLIR()} to ${out.type.toMLIR()}"
                    }

                    Prims.Comp.TRANSPOSE -> {
                        val dest = instr.args[0]
                        val arr = instr.args[1]
                        val arrTy = arr.type as ArrayType

                        // produces something like "(a, b) -> (b, a)"
                        val namMap = List(arrTy.shape.size) { 'a' + it }
                            .let {
                                val a = it.joinToString(prefix = "(", postfix = ")")
                                val b = it.reversed().joinToString(prefix = "(", postfix = ")")
                                "$a -> $b"
                            }

                        //  affine_map<(d0, d1, d2)[s0, s1] -> (d0 + s0 + d1 * 2 + d2 * s1)>

                        // produces something like "affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>"
                        val affineMap = let {
                            val a = List(arrTy.shape.size) { "d$it" }.joinToString(prefix = "(", postfix = ")")
                            val b = List(arrTy.shape.size - 1) { "s$it" }.joinToString(prefix = "[", postfix = "]")
                            val c = List(arrTy.shape.size) {
                                var r = "d$it"
                                if (it > 0) {
                                    r += " * s${it-1}"
                                }
                                r
                            }.reduce { aa, ab -> "$aa + $ab" }
                            "affine_map<$a$b -> ($c)>"
                        }

                        val transposed = newVar().asMLIR()
                        val traTy = Ty.memref(arrTy.shape.reversed(), arrTy.inner.toMLIR(), arrTy.vaOff, ", $affineMap")
                        body += "$transposed = memref.transpose ${arr.asMLIR()} $namMap : ${arrTy.toMLIR()} to $traTy"

                        body += "memref.copy ${transposed}, ${dest.asMLIR()} : $traTy to ${dest.type.toMLIR()}"
                    }

                    else -> error("primitive ${instr.instr.id} not implemented")
                }

                is CommentInstr -> {}

                else -> error("$instr not implemented")
            }
        }.onFailure {
            log("in mlir emit instr $uid:$idx: ${it.message}")
            throw it
        }
    }

    return body
}

fun IrBlock.emitMLIRFinalize(body: List<String>): String =
    function(
        name,
        private,
        args.mapTo(mutableListOf()) { it.asMLIR() to it.type.toMLIR() },
        rets.map { it.asMLIR() to it.type.toMLIR() },
        body
    )
