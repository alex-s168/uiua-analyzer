package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun IrBlock.emitMLIR(): String {
    fun IrVar.asMLIR(): MLIRVar =
        "%${id}"

    val body = mutableListOf<String>()

    fun castIfNec(variable: IrVar, want: Type): IrVar  =
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
        body += Inst.add(
            dest = outs[0].asMLIR(),
            type = outs[0].type.toMLIR(wantTensor = true),
            a = castIfNec(args[0], outs[0].type).asMLIR(),
            b = castIfNec(args[1], outs[0].type).asMLIR(),
            float = outs[0].type == Types.double
        )
    }

    instrs.forEach { instr ->
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                "ADD" -> instr.binary(Inst::add)
                "SUB" -> instr.binary(Inst::sub)
                "MUL" -> instr.binary(Inst::mul)
                "DIV" -> instr.binary(Inst::div)
                "PRIMES" -> TODO()

                "BOX" -> TODO()
                "UN_BOX" -> TODO()

                "EACH" -> TODO()

                else -> error("")
            }
            else -> TODO()
        }
    }

    return function(
        name,
        args.map { it.asMLIR() to it.type.toMLIR() },
        rets.map { it.asMLIR() to it.type.toMLIR() },
        body
    )
}