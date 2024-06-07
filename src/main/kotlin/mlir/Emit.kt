package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun IrBlock.emitMLIR(): String {
    fun IrVar.asMLIR(): MLIRVar =
        "%${id}"

    val body = mutableListOf<String>()

    fun castIfNec(body: MutableList<String>, variable: IrVar, want: Type): IrVar =
        if (want is ArrayType && variable.type !is ArrayType)
            castIfNec(body, variable, want.inner)
        else
            if (variable.type == want) variable
            else {
                val new = newVar().copy(type = want)
                body.add(castInstr(
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
    ) {
        val outTy = outs[0].type
        if (outTy is ArrayType) {
            val arrArg = args.find { it.type is ArrayType }
            val cast = args.map {
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
                castIfNec(body, args[0], outTy).asMLIR(),
                castIfNec(body, args[1], outTy).asMLIR(),
                outTy == Types.double
            )
        }
    }

    fun callWithOptFill(dests: List<MLIRVar>, types: List<MLIRType>, fill: IrVar?, fn: IrBlock, vararg args: MLIRVar): String =
        if (fn.fillArg != null) {
            Inst.call(
                dests,
                types,
                fn.asMLIR(),
                *(arrayOf(fill!!.asMLIR()) + args)
            )
        } else {
            Inst.call(
                dests,
                types,
                fn.asMLIR(),
                *args
            )
        }

    fun callWithOptFill(dests: List<MLIRVar>, types: List<MLIRType>, fn: IrBlock, vararg args: MLIRVar) =
        callWithOptFill(dests, types, fillArg, fn, *args)

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
            is PrimitiveInstr -> when (instr.instr.id) {
                "ADD" -> instr.binary(body, Inst::add, Inst::arrAdd)
                "SUB" -> instr.binary(body, Inst::sub, Inst::arrSub)
                "MUL" -> instr.binary(body, Inst::mul, Inst::arrMul)
                "DIV" -> instr.binary(body, Inst::div, Inst::arrDiv)

                "PRIMES" -> {
                    body += Inst.call(
                        dests = listOf(instr.outs[0].asMLIR()),
                        types = listOf(instr.outs[0].type.toMLIR()), // should be memref<? x i64>
                        MLIRFn("_\$_rt_primes", listOf(Ty.int(64))),
                        castIfNec(body, instr.args[0], Types.int).asMLIR()
                    )
                }

                Prim.Comp.BOX_CREATE -> {
                    body += Inst.memRefAlloc(
                        instr.outs[0].asMLIR(),
                        instr.outs[0].type.toMLIR()
                    )
                }

                Prim.Comp.BOX_STORE -> {
                    val idx = newVar().asMLIR()
                    body += Inst.constant(idx, Ty.index, "0")

                    body += Inst.memRefStore(
                        instr.args[0].type.toMLIR(),
                        instr.args[1].asMLIR(),
                        instr.args[0].asMLIR(),
                        idx
                    )
                }

                Prim.Comp.BOX_LOAD -> {
                    val type = instr.args[0].type

                    val idx = newVar().asMLIR()
                    body += Inst.constant(idx, Ty.index, "0")

                    body += Inst.memRefLoad(
                        instr.outs[0].asMLIR(),
                        type.toMLIR(),
                        instr.args[0].asMLIR(),
                        idx
                    )
                }

                Prim.Comp.BOX_DESTROY -> {
                    body += Inst.memRefDealloc(
                        instr.args[0].asMLIR(),
                        instr.args[0].type.toMLIR()
                    )
                }

                Prim.Comp.REPEAT -> {
                    val times = instr.args[0].asMLIR()
                    val (_, fnd) = funDeclFor(args[1])

                    val counter = newVar()

                    val inner = mutableListOf<String>()
                    val arg = castIfNec(
                        inner,
                        counter,
                        fnd.args[0].type
                    )

                    inner += callWithOptFill(
                        listOf(),
                        listOf(),
                        fnd,
                        arg.asMLIR()
                    )

                    body += Inst.affineParallelFor(
                        listOf(counter.asMLIR()),
                        listOf("0"),
                        listOf(times),
                        inner
                    )
                }

                Prim.Comp.DIM -> {
                    body += Inst.memRefDim(
                        dest = instr.outs[0].asMLIR(),
                        memRefType = instr.args[0].type.toMLIR(),
                        memRef = instr.args[0].asMLIR(),
                        dim = instr.args[1].asMLIR()
                    )
                }

                Prim.EACH -> {
                    val src = instr.args[1]
                    val srcTy = src.type as ArrayType
                    val (_, fnd) = funDeclFor(instr.args[0])

                    val mShape = emitShapeOf(body, src)

                    val iterCoords = List(mShape.size) { newVar().asMLIR() }
                    val iterRes = newVar().asMLIR()

                    val innerBody = mutableListOf<String>()

                    val srcElem = newVar().asMLIR()
                    innerBody += Inst.memRefLoad(
                        dest = srcElem,
                        memRefType = srcTy.toMLIR(),
                        memRef = src.asMLIR(),
                        *iterCoords.toTypedArray()
                    )

                    innerBody += callWithOptFill(
                        dests = listOf(iterRes),
                        types = listOf(fnd.rets[0].type.toMLIR()),
                        fn = fnd,
                        srcElem
                    )

                    body += Inst.Compound.memRefGenerate(
                        dest = instr.outs[0].asMLIR(),
                        memRefTy = Ty.memref(src.type.shape, fnd.rets[0].type.toMLIR()),
                        allDims = mShape,
                        dynDims = mShape.filterIndexed { idx, _ -> srcTy.shape[idx] == -1 },
                        iterCords = iterCoords,
                        iterRes = iterRes,
                        iterResTy = fnd.rets[0].type.toMLIR(),
                        inner = innerBody
                    )
                }

                Prim.ROWS -> {
                    TODO()
                }

                Prim.REDUCE -> {
                    TODO("implement reduce using box (memref<1xT>) as acc")
                }

                Prim.FILL -> {
                    val (_, fillValFn) = funDeclFor(instr.args[0])
                    val (_, opFn) = funDeclFor(instr.args[1])
                    val opArgs = instr.args.drop(2)

                    val fillVal = newVar()
                    body += callWithOptFill(
                        dests = listOf(fillVal.asMLIR()),
                        types = listOf(fillValFn.rets[0].type.toMLIR()),
                        fillValFn,
                    )

                    body += callWithOptFill(
                        dests = instr.outs.map { it.asMLIR() },
                        types = instr.outs.map { it.type.toMLIR() },
                        fill = fillVal,
                        opFn,
                        *opArgs.map { it.asMLIR() }.toTypedArray()
                    )
                }

                else -> error("")
            }
            is PushFnRefInstr -> {} // ignore
            else -> error("$instr not implemented")
        }
    }

    val mArgs = args.mapTo(mutableListOf()) { it.asMLIR() to it.type.toMLIR() }
    fillArg?.let {
        mArgs.add(0, it.asMLIR() to it.type.toMLIR())
    }

    return function(
        name,
        mArgs,
        rets.map { it.asMLIR() to it.type.toMLIR() },
        body
    )
}