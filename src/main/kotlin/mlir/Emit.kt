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

    fun IrInstr.binary(op: (dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) -> String) {
        body += op(
            outs[0].asMLIR(),
            outs[0].type.toMLIR(),
            castIfNec(args[0], outs[0].type).asMLIR(),
            castIfNec(args[1], outs[0].type).asMLIR(),
            outs[0].type == Types.double
        )
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

    instrs.forEach { instr ->
        when (instr.instr) {
            is NumImmInstr -> {
                val value = instr.instr.value.toString()
                body += Inst.constant(instr.outs[0].asMLIR(), instr.outs[0].type.toMLIR(), value)
            }
            is PrimitiveInstr -> when (instr.instr.id) {
                "ADD" -> instr.binary(Inst::add)
                "SUB" -> instr.binary(Inst::sub)
                "MUL" -> instr.binary(Inst::mul)
                "DIV" -> instr.binary(Inst::div)
                "PRIMES" -> {
                    body += Inst.call(
                        dest = instr.outs[0].asMLIR(),
                        type = instr.outs[0].type.toMLIR(), // should be memref<? x i64>
                        MLIRFn("_\$_rt_primes", listOf(Ty.int(64))),
                        castIfNec(instr.args[0], Types.int).asMLIR()
                    )
                }

                "BOX" -> TODO()
                "UN_BOX" -> TODO()

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