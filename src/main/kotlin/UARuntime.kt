package me.alex_s168.uiua

object UARuntime {
    data class Func(
        val name: String,
        val type: FnType
    )

    val panic = Func(
        "_\$_rt_panic",
        Types.func(listOf(Types.int, Types.int), listOf())
    )
}