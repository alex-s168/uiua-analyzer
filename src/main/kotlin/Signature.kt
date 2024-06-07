package me.alex_s168.uiua

import me.alex_s168.uiua.ast.AstNode

// combine signatures as if both were used on the stack after each other
operator fun Signature.plus(other: Signature): Signature {
    val totalInputs = this.inputs + other.inputs - minOf(this.outputs, other.inputs)
    val totalOutputs = this.outputs + other.outputs - minOf(this.outputs, other.inputs)
    return Signature(totalInputs, totalOutputs)
}

internal fun signature(
    instr: Instr,
    onStack: (Int) -> AstNode,
): Signature {
    if (instr is ImmInstr)
        return Signature(0, 1)

    if (instr is CommentInstr || instr is FlagInstr)
        return Signature(0, 0)

    require(instr is PrimitiveInstr)

    return when (instr.id) {
        Prim.ADD -> Signature(2, 1)
        Prim.SUB -> Signature(2, 1)
        Prim.MUL -> Signature(2, 1)
        Prim.DIV -> Signature(2, 1)

        Prim.LEN -> Signature(1, 1)

        Prim.PRIMES -> Signature(1, 1)

        Prim.BOX -> Signature(1, 1)
        Prim.UN_BOX -> Signature(1, 1)

        Prim.POP -> Signature(1, 0)
        Prim.DUP -> Signature(1, 2)
        Prim.FLIP -> Signature(2, 2)

        Prim.EACH -> Signature(2, 1)
        Prim.REDUCE -> {
            val fn = onStack(0).value.getA().instr as PushFnInstr
            Signature(fn.fn.signature.inputs, 1)
        }
        Prim.ROWS -> {
            val each = onStack(0).value.getA().instr as PushFnInstr
            each.fn.signature.mapIns { it + 1 } // arg 0 is also part of the inst
        }
        Prim.FILL -> {
            // onStack(0) is the fill value
            val fn = onStack(1).value.getA().instr as PushFnInstr
            fn.fn.signature.mapIns { it + 2 } // arg 0 & arg 1
        }

        else -> error("Unknown primitive instruction ${instr.id}!")
    }
}