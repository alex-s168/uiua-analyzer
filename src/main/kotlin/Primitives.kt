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
        val ARR_CLONE = prim("cARR_CLONE")
        val ARR_COPY = prim("cARR_COPY") // [dest], [src]
        val ARR_DESTROY = prim("cARR_DESTROY")

        val REPEAT = prim("cREPEAT") // [start], [end], [fn which takes counter], [additional]...
        val DIM = prim("cDIM") // [arr], [dim]
        val COUNT_NOTZERO = prim("cCOUNT_NOTZERO") // [arr]

        val SINK = prim("cSINK") // [val]

        val RESHAPE_VIEW = prim("cRESHAPE_VIEW") // [sha], [arr]

        /** outputs can be anything as program won't continue after this */
        val PANIC = prim("cPANIC")
    }

    object Front {
        val REDUCE_DEPTH = prim("fREDUCE_DEPTH")
    }

    val ADD = prim("ADD")
    val SUB = prim("SUB")
    val MUL = prim("MUL")
    val DIV = prim("DIV")
    val POW = prim("POW")
    val MAX = prim("MAX")
    val LT = prim("LT")
    val EQ = prim("EQ")

    // TODO: investigate if these are the real names
    val LEN = prim("LEN")
    val SWITCH = prim("SWITCH") // [conds: arg list], [dests: arg lists], on: int, [args]...     the last case is always default case
    val CALL = prim("CALL")

    val PRIMES = prim("PRIMES")
    val RANGE = prim("RANGE")

    val BOX = prim("BOX")
    val UN_BOX = prim("UN_BOX")

    val POP = prim("POP")
    val DUP = prim("DUP")
    val FLIP = prim("FLIP")
    val OVER = prim("OVER")

    val EACH = prim("EACH") // [fn], [arrays]...
    val REDUCE = prim("REDUCE")
    val ROWS = prim("ROWS")
    val REVERSE = prim("REVERSE")
    val FIX = prim("FIX")
    val SHAPE = prim("SHAPE")
    val PICK = prim("PICK")
    val UNDO_PICK = prim("UNDO_PICK")
    val TABLE = prim("TABLE")
    val RESHAPE = prim("RESHAPE") // [shape: arg list], [array]
    val UN_SHAPE = prim("UN_SHAPE")
    val WHERE = prim("WHERE")
    val NOW = prim("NOW")
    val JOIN = prim("JOIN")

    val FILL = prim("FILL") // [fill provider], [fn], [args]...
}