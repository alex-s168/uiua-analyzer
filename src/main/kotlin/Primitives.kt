package me.alex_s168.uiua

object Prim {
    val ALL = mutableSetOf<String>()

    private fun prim(name: String): String =
        name.also { ALL.add(it) }

    object Comp {
        val USE = prim("cUSE")

        val BOX_CREATE = prim("cBOX_CREATE")
        val BOX_STORE = prim("cBOX_STORE") // [dest], [val]
        val BOX_LOAD = prim("cBOX_LOAD")
        val BOX_DESTROY = prim("cBOX_DESTROY")

        /** array that is only meant to be used as args for other comp primitives */
        val ARG_ARR = prim("cARG_ARR") // [elem]...

        val ARR_MATERIALIZE = prim("cARR_MAT") // [src: arg_arr]
        val ARR_ALLOC = prim("cARR_ALLOC") // [shape: arg_arr]
        val ARR_STORE = prim("cARR_STORE") // [dest], [indecies: arg_arr], [value]
        val ARR_LOAD = prim("cARR_LOAD") // [arr], [indecies: arr_arg]
        val ARR_DESTROY = prim("cARR_DESTROY")

        val REPEAT = prim("cREPEAT") // [times], [fn which takes counter]
        val DIM = prim("cDIM") // [arr], [dim]
    }

    val ADD = prim("ADD")
    val SUB = prim("SUB")
    val MUL = prim("MUL")
    val DIV = prim("DIV")

    val PRIMES = prim("PRIMES")

    val BOX = prim("BOX")
    val UN_BOX = prim("UN_BOX")

    val POP = prim("POP")
    val DUP = prim("DUP")
    val FLIP = prim("FLIP")

    val EACH = prim("EACH")
    val REDUCE = prim("REDUCE")
    val ROWS = prim("ROWS")

    const val FILL = "FILL"
}