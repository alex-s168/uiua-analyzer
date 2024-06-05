package me.alex_s168.uiua

// combine signatures as if both were used on the stack after each other
operator fun Signature.plus(other: Signature): Signature {
    val totalInputs = this.inputs + other.inputs - minOf(this.outputs, other.inputs)
    val totalOutputs = this.outputs + other.outputs - minOf(this.outputs, other.inputs)
    return Signature(totalInputs, totalOutputs)
}

// TODO: make more advanced for ROWS and others
fun signature(instr: Instr): Signature {
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

        "EACH" -> Signature(2, 1)
        else -> error("Unknown primitive instruction ${instr.id}!")
    }
}