package me.alex_s168.uiua

import me.alex_s168.uiua.ast.AstInstrNode
import me.alex_s168.uiua.ast.AstNode

// combine signatures as if both were used on the stack after each other
operator fun Signature.plus(other: Signature): Signature {
    val totalInputs = this.inputs + other.inputs - minOf(this.outputs, other.inputs)
    val totalOutputs = this.outputs + other.outputs - minOf(this.outputs, other.inputs)
    return Signature(totalInputs, totalOutputs)
}

// TODO: make more advanced for ROWS and others
fun signature(
    instr: Instr,
    directArgs: (Int) -> Instr,
    onStack: (Int) -> AstNode,
): Signature {
    if (instr is ImmInstr)
        return Signature(0, 1)

    if (instr is CommentInstr || instr is FlagInstr)
        return Signature(0, 0)

    require(instr is PrimitiveInstr)

    return when (instr.id) {
        "cUSE" -> Signature(1, 1)

        "ADD" -> Signature(2, 1)
        "SUB" -> Signature(2, 1)
        "MUL" -> Signature(2, 1)
        "DIV" -> Signature(2, 1)
        "PRIMES" -> Signature(1, 1)

        "BOX" -> Signature(1, 1)
        "UN_BOX" -> Signature(1, 1)

        "POP" -> Signature(1, 0)
        "DUP" -> Signature(1, 2)
        "FLIP" -> Signature(2, 2)

        "EACH" -> Signature(2, 1)
        "REDUCE" -> Signature(2, 1)
        "ROWS" -> {
            val each = onStack(0).value.getA().instr as PushFnInstr
            each.fn.signature.mapIns { it + 1 } // arg 0 is also part of the inst
        }

        "FILL" -> {
            // onStack(0) is the fill value
            val fn = onStack(1).value.getA().instr as PushFnInstr
            fn.fn.signature.mapIns { it + 2 } // arg 0 & arg 1
        }

        else -> error("Unknown primitive instruction ${instr.id}!")
    }
}