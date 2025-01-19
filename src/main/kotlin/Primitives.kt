package me.alex_s168.uiua

typealias Prim = Short

object Prims {
    val all = mutableMapOf<Prim, String>()
    val all2 = mutableMapOf<String, Prim>()
    private var nextPrimId: Prim = 0

    private fun prim(name: String): Prim =
        (nextPrimId ++)
            .also { all[it] = name }
            .also { all2[name] = it }

    object Comp {
        val INVALID = prim("?????")
        val USE = prim("cUSE")

        val BOX_CREATE = prim("cBOX_CREATE")
        val BOX_STORE = prim("cBOX_STORE") // [dest], [val]
        val BOX_LOAD = prim("cBOX_LOAD")
        val BOX_DESTROY = prim("cBOX_DESTROY")

        /** array that is only meant to be used as args for other comp primitives */
        val ARG_ARR = prim("cARG_ARR") // [elem]...

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

        val OFF_VIEW_1D = prim("cOFF_VIEW") //[arr], [begin idx], [len]
        val RESHAPE_VIEW = prim("cRESHAPE_VIEW") // [sha: 1d array], [arr]
        val FIX_ARR = prim("cFIX_ARR") // [arr], [arr original dimension]...

        val RT_EXTEND_SCALAR = prim("cRT_EXTEND_SCALAR") // [base], [targetLen], [fillWith]
        val RT_EXTEND_REPEAT = prim("cRT_EXTEND_REPEAT") // [base], [targetLen], [extendWithRep]

        /** outputs can be anything as program won't continue after this */
        val PANIC = prim("cPANIC")

        val UNDEF = prim("cUNDEF")

        val DYN_TYPEID = prim("cDYN_TYPEID") // [prim]
        val DYN_UNWRAP = prim("cDYN_UNWRAP") // has typeParam
        val DYN_WRAP = prim("cDYN_WRAP") // [value]
        val DYN_FREE = prim("cDYN_FREE") // [dyn]

        val TRANSPOSE = prim("cTRANSPOSE") // [dest], [src]

        // the following are identical to the non-P versions, but don't have the output variable in the outs, so that analysis functions don't know that these are the origin of vars. this is only used for generating complicated code that should be generated during emit mlir, but are not
        val EMIT_ARR_ALLOC_P = prim("ceARR_ALLOC_P") // [out var], [shape]
    }

    object Front {
        val REDUCE_DEPTH = prim("fREDUCE_DEPTH")
        val UN_TRANSPOSE = prim("fUN_TRANSPOSE")
    }

    val ADD = prim("ADD")
    val SUB = prim("SUB")
    val MUL = prim("MUL")
    val DIV = prim("DIV")
    val POW = prim("POW")
    val MAX = prim("MAX")
    val MIN = prim("MIN")
    val LT = prim("LT")
    val EQ = prim("EQ")
    val MOD = prim("MOD")
    val ABS = prim("ABS")
    val SIN = prim("SIN")
    val NEG = prim("NEG")
    val SQRT = prim("SQRT")
    val ASIN = prim("ASIN")
    val FLOOR = prim("FLOOR")
    val CEIL = prim("CEIL")
    val ROUND = prim("ROUND")

    val RAND = prim("RAND")
    val REPLACE_RAND = prim("REPLACE_RAND") // pop and then rand
    val COMPLEX = prim("COMPLEX")

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
    val RERANK = prim("RERANK") // [rank: int], [array]
    val UNDO_RERANK = prim("UNDO_RERANK")
    val UN_SHAPE = prim("UN_SHAPE")
    val WHERE = prim("WHERE")
    val NOW = prim("NOW")
    val JOIN = prim("JOIN")
    val DESHAPE = prim("DESHAPE")
    val KEEP = prim("KEEP")
    val IDENTITY = prim("IDENTITY")
    val UN_COUPLE = prim("UN_COUPLE") // [arr]
    val TRANSPOSE = prim("TRANSPOSE")

    val FILL = prim("FILL") // [fill provider], [fn], [args]...
}
