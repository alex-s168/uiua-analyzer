package me.alex_s168.uiua

import me.alex_s168.uiua.mlir.legalizeMLIR

object UARuntime {
    data class Func(
        val name: String,
        val type: FnType
    )

    fun templated(name: String, retProv: (argTypes: List<Type>) -> List<Type>) =
        { argTypes: List<Type> -> Func(
            "${name}_${ argTypes.joinToString(separator = "_").legalizeMLIR() }",
            FnType(null, argTypes, retProv(argTypes))
        ) }


    fun autobyteToByte(it: Type) =
        if (it == Types.autobyte) Types.byte else it

    fun autobyteToByte(li: List<Type>) =
        li.map { autobyteToByte(it) }

    fun autobyteToByte(fn: (List<Type>) -> Func) =
        { types: List<Type> -> fn(autobyteToByte(types)) }

    // =========================================================================

    val panic = Func(
        "_\$_rt_panic",
        Types.func(listOf(Types.int, Types.int, Types.int, Types.int), listOf())
    )

    val time = Func(
        "_\$_rt_time",
        Types.func(listOf(), listOf(Types.double))
    )

    // [base], [targetLen], [fillWith]
    val extendScalar = autobyteToByte {  Func(
        "_\$_rt_extendScalar_${(it[0] as ArrayType).of}",
        Types.func(it, listOf(it[0]))
    ) }

    // [base], [targetLen], [extendWithRep]
    val extendRepeat = autobyteToByte { Func(
        "_\$_rt_extendRepeat_${(it[0] as ArrayType).of}",
        Types.func(it, listOf(it[0]))
    ) }
}