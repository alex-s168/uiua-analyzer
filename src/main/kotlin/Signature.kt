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
        Prims.ADD -> Signature(2, 1)
        Prims.SUB -> Signature(2, 1)
        Prims.MUL -> Signature(2, 1)
        Prims.DIV -> Signature(2, 1)
        Prims.POW -> Signature(2, 1)
        Prims.EQ -> Signature(2, 1)
        Prims.MOD -> Signature(2, 1)
        Prims.ABS -> Signature(1, 1)
        Prims.SIN -> Signature(1, 1)
        Prims.NEG -> Signature(1, 1)
        Prims.SQRT -> Signature(1, 1)
        Prims.ASIN -> Signature(1, 1)
        Prims.FLOOR -> Signature(1, 1)
        Prims.CEIL -> Signature(1, 1)
        Prims.ROUND -> Signature(1, 1)
        Prims.RERANK -> Signature(2, 1)
        Prims.UNDO_RERANK -> Signature(3, 1)
        Prims.UN_COUPLE -> Signature(1, 2)

        Prims.SHAPE -> Signature(1, 1)
        Prims.FIX -> Signature(1, 1)
        Prims.LEN -> Signature(1, 1)

        Prims.PRIMES -> Signature(1, 1)
        Prims.RANGE -> Signature(1, 1)

        Prims.BOX -> Signature(1, 1)
        Prims.UN_BOX -> Signature(1, 1)

        Prims.POP -> Signature(1, 0)
        Prims.DUP -> Signature(1, 2)
        Prims.FLIP -> Signature(2, 2)
        Prims.OVER -> Signature(2, 3)

        Prims.EACH -> {
            val each = onStack(0).value.a!!.instr as PushFnInstr
            each.fn.signature!!.mapIns { it + 1 }
        }
        Prims.REDUCE, Prims.Front.REDUCE_DEPTH -> {
            val fn = onStack(0).value.a!!.instr as PushFnInstr
            Signature(fn.fn.signature!!.inputs, 1)
        }
        Prims.ROWS -> {
            val each = onStack(0).value.a!!.instr as PushFnInstr
            each.fn.signature!!.mapIns { it + 1 } // arg 0 is also part of the inst
        }
        Prims.FILL -> {
            // onStack(0) is the fill value
            val fn = onStack(1).value.a!!.instr as PushFnInstr
            fn.fn.signature!!.mapIns { it + 2 } // arg 0 & arg 1
        }
        Prims.TABLE -> {
            val inner = onStack(0).value.a!!.instr as PushFnInstr
            require(inner.fn.signature!!.inputs >= 2) {
                "function passed to table needs to take in 2 or more arguments"
            }
            inner.fn.signature.mapIns { it + 1 } // arg 0 is also part of the inst
        }
        Prims.WHERE -> Signature(1, 1)
        Prims.REVERSE -> Signature(1, 1)
        Prims.PICK -> Signature(2, 1)
        Prims.UNDO_PICK -> Signature(3, 1)
        Prims.RESHAPE -> Signature(2, 1)
        Prims.UN_SHAPE -> Signature(1, 1)
        Prims.NOW -> Signature(0, 1)
        Prims.JOIN -> Signature(2, 1)
        Prims.DESHAPE -> Signature(1, 1)
        Prims.KEEP -> Signature(2, 1)
        Prims.RAND -> Signature(0, 1)
        Prims.REPLACE_RAND -> Signature(1, 1)
        Prims.COMPLEX -> Signature(2, 1)
        Prims.IDENTITY -> Signature(1, 1)
        Prims.TRANSPOSE -> Signature(1, 1)
        Prims.Front.UN_TRANSPOSE -> Signature(1, 1)

        Prims.CALL -> {
            val fn = onStack(0).value.a!!.instr as PushFnInstr
            fn.fn.signature!!.mapIns { it + 1 } // arg 0 is also part of the inst
        }

        else -> error("Unknown primitive instruction ${instr.id}!")
    }
}