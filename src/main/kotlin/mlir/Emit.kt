package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

/*
TODO emit:
 linalg.add ins(%5, %3 : i64, memref<? x i64>)              %3 is input array (even though marked with :i64)
             outs(%3 : memref<? x i64>)
 */

fun IrBlock.emitMLIR(): String {
    fun IrVar.asMLIR(): MLIRVar =
        "%${id}"

    val body = mutableListOf<String>()

    fun castIfNec(variable: IrVar, want: Type): IrVar =
        if (want is ArrayType && variable.type !is ArrayType)
            castIfNec(variable, want.inner)
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


    fun emitShapeOf(src: IrVar): List<MLIRVar> {
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
        op: (dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) -> String,
        opArr: (dest: MLIRVar, destType: MLIRType, sources: List<Pair<MLIRVar, MLIRType>>) -> String,
    ) {
        val outTy = outs[0].type
        if (outTy is ArrayType) {
            val arrArg = args.find { it.type is ArrayType }
            val cast = args.map {
                if (it.type is ArrayType) it
                else castIfNec(it, outTy)
            }

            val mDynShape = arrArg?.let {
                val mShape = emitShapeOf(it)
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
                castIfNec(args[0], outTy).asMLIR(),
                castIfNec(args[1], outTy).asMLIR(),
                outTy == Types.double
            )
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
                "ADD" -> instr.binary(Inst::add, Inst::arrAdd)
                "SUB" -> instr.binary(Inst::sub, Inst::arrSub)
                "MUL" -> instr.binary(Inst::mul, Inst::arrMul)
                "DIV" -> instr.binary(Inst::div, Inst::arrDiv)
                "PRIMES" -> {
                    body += Inst.call(
                        dest = instr.outs[0].asMLIR(),
                        type = instr.outs[0].type.toMLIR(), // should be memref<? x i64>
                        MLIRFn("_\$_rt_primes", listOf(Ty.int(64))),
                        castIfNec(instr.args[0], Types.int).asMLIR()
                    )
                }

                "BOX" -> {
                    val type = instr.outs[0].type

                    body += Inst.memRefAlloc(
                        instr.outs[0].asMLIR(),
                        type.toMLIR()
                    )

                    val idx = newVar().asMLIR()
                    body += Inst.constant(idx, Ty.index, "0")

                    body += Inst.memRefStore(
                        type.toMLIR(),
                        instr.args[0].asMLIR(),
                        instr.outs[0].asMLIR(),
                        idx
                    )
                }
                "UN_BOX" -> {
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
                "EACH" -> {
                    val src = instr.args[1]
                    val srcTy = src.type as ArrayType
                    val fn = (instrDeclFor(instr.args[0])!!.instr as PushFnRefInstr).fn
                    val fnd = ref(fn)!!

                    val mShape = emitShapeOf(src)

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

                    innerBody += Inst.call(
                        dest = iterRes,
                        type = fnd.rets[0].type.toMLIR(),
                        fn = fnd.asMLIR(),
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

                else -> error("")
            }
            is PushFnRefInstr -> {} // ignore
            else -> error("$instr not implemented")
        }
    }

    return function(
        name,
        args.map { it.asMLIR() to it.type.toMLIR() },
        rets.map { it.asMLIR() to it.type.toMLIR() },
        body
    )
}