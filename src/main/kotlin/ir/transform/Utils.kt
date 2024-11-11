package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun genCallBlockFnTail(innerFnType: FnType, ref: Map<BlockId, IrBlock>): IrBlock {
    return IrBlock(anonFnName(), ref).apply {
        val outs = innerFnType.rets.map {
            newVar().copy(type = it)
                .also(rets::add)
        }

        val ar = innerFnType.args.map {
            newVar().copy(type = it)
                .also(args::add)
        }

        val fn = newVar().copy(type = innerFnType)
            .also(args::add)

        instrs += IrInstr(
            outs.toMutableList(),
            PrimitiveInstr(Prims.CALL),
            (listOf(fn) + ar).toMutableList()
        )
    }
}

fun IrVar.genFix(put: (IrInstr) -> Unit, newVar: () -> IrVar): IrVar {
    val out = newVar()
        .copy(type = Types.array(this.type))

    put(IrInstr(
        mutableListOf(out),
        PrimitiveInstr(Prims.FIX),
        mutableListOf(this)
    ))

    return out
}

fun List<IrVar>.wrapInArgArray(newVar: () -> IrVar, type: Type = first().type, put: (IrInstr) -> Unit): IrVar {
    val v = newVar().copy(type = Types.array(type, size))
    put(IrInstr(
        mutableListOf(v),
        PrimitiveInstr(Prims.Comp.ARG_ARR),
        toMutableList()
    ))
    return v
}

fun IrVar.into(dest: IrVar, put: (IrInstr) -> Unit) {
    if (this.id != dest.id) {
        put(IrInstr(
            mutableListOf(dest),
            PrimitiveInstr(Prims.Comp.USE),
            mutableListOf(this)
        ))
    }
}

fun constants(newVar: () -> IrVar, vararg const: Double, type: Type = Types.double, put: (IrInstr) -> Unit): List<IrVar> =
    const.map {
        val constant = newVar().copy(type = type)
        put(IrInstr(
            mutableListOf(constant),
            NumImmInstr(it),
            mutableListOf()
        ))
        constant
    }

fun constant(const: Double, type: Type = Types.double, newVar: () -> IrVar, put: (IrInstr) -> Unit): IrVar =
    constants(newVar, const, type=type, put=put)[0]

fun constantArr(newVar: () -> IrVar, vararg const: Double, type: Type = Types.double, put: (IrInstr) -> Unit): IrVar =
    constants(newVar, *const, type = type, put = put).wrapInArgArray(newVar, type, put)

fun constantArr(dest: IrVar, newVar: () -> IrVar, vararg const: Double, type: Type = Types.double, put: (IrInstr) -> Unit) {
    constantArr(newVar, *const, type = type, put = put).into(dest, put)
}

fun oneDimLoad(dest: IrVar, arr: IrVar, newVar: () -> IrVar, idx: Int, put: (IrInstr) -> Unit) {
    val indecies = constantArr(newVar, idx.toDouble(), type = Types.int, put = put)

    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prims.Comp.ARR_LOAD),
        mutableListOf(arr, indecies)
    ))
}

fun oneDimLoad(arr: IrVar, newVar: () -> IrVar, idx: Int, put: (IrInstr) -> Unit): IrVar {
    val variable = newVar().copy(type = (arr.type as ArrayType).of.makeVaOffIfArray())
    oneDimLoad(dest = variable, arr = arr, newVar = newVar, idx = idx, put = put)
    return variable
}

fun oneDimLoad(dest: IrVar, arr: IrVar, newVar: () -> IrVar, idx: IrVar, put: (IrInstr) -> Unit) {
    val indecies = listOf(idx).wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prims.Comp.ARR_LOAD),
        mutableListOf(arr, indecies)
    ))
}

fun oneDimLoad(arr: IrVar, newVar: () -> IrVar, idx: IrVar, put: (IrInstr) -> Unit): IrVar {
    val variable = newVar().copy(type = (arr.type as ArrayType).of.makeVaOffIfArray())
    oneDimLoad(dest = variable, arr = arr, newVar = newVar, idx = idx, put = put)
    return variable
}

// TODO: make primitive!
fun oneDimFillLoad(
    dest: IrVar,
    fill: IrVar?,
    arr: IrVar,
    putBlock: (IrBlock) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    idx: IrVar,
    put: (IrInstr) -> Unit
) {
    put(comment("+++ one dim fill load"))

    val arrTy = arr.type as ArrayType

    if (fill == null && !unfilledLoadBoundsCheck) {
        oneDimLoad(dest, arr, newVar, idx, put)
    }
    else {
        val err = IrBlock(anonFnName(), ref).apply {
            fillArg = fill?.let { newVar().copy(type = it.type) }

            val arr = newVar().copy(type = arrTy).also { args += it }
            val idx = newVar().copy(type = idx.type).also { args += it }

            val value = newVar().copy(type = arrTy.of.makeVaOffIfArray()).also { rets += it }

            if (fillArg == null) {
                instrs += IrInstr(
                    mutableListOf(value),
                    PrimitiveInstr(Prims.Comp.PANIC),
                    mutableListOf()
                )
            } else {
                fillArg!!.into(value) { instrs += it }
            }

            putBlock(this)
        }

        val ok = IrBlock(anonFnName(), ref).apply {
            val arr = newVar().copy(type = arrTy).also { args += it }
            val idx = newVar().copy(type = idx.type).also { args += it }

            val value = newVar().copy(type = arrTy.of.makeVaOffIfArray()).also { rets += it }

            val idc = listOf(idx).wrapInArgArray(::newVar) { instrs += it }

            instrs += IrInstr(
                mutableListOf(value),
                PrimitiveInstr(Prims.Comp.ARR_LOAD),
                mutableListOf(arr, idc)
            )

            putBlock(this)
        }

        val len = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(len),
            PrimitiveInstr(Prims.LEN),
            mutableListOf(arr)
        ))

        val lt = newVar().copy(type = Types.byte)
        put(IrInstr(
            mutableListOf(lt),
            PrimitiveInstr(Prims.LT),
            mutableListOf(len, idx) // idx < len
        ))

        val (zero, one) = constants(newVar, 0.0, 1.0, type = Types.size, put = put)

        switch(
            listOf(dest),
            newVar,
            lt,
            listOf(arr, idx),
            zero to err,
            one to ok,
            put = put,
        )
    }

    put(comment("--- one dim fill load"))
}

fun oneDimFillLoad(
    fill: IrVar?,
    arr: IrVar,
    putBlock: (IrBlock) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    idx: IrVar,
    put: (IrInstr) -> Unit
): IrVar {
    val variable = newVar().copy(type = (arr.type as ArrayType).of.makeVaOffIfArray())
    oneDimFillLoad(dest = variable, fill = fill, arr = arr, putBlock = putBlock, ref = ref, newVar = newVar, idx = idx, put = put)
    return variable
}

fun switch(
    dest: List<IrVar>,
    newVar: () -> IrVar,
    on: IrVar,
    inputs: List<IrVar>,
    vararg cases: Pair<IrVar, IrBlock>,
    put: (IrInstr) -> Unit
) {
    val conds = cases.map { it.first }
        .wrapInArgArray(newVar, put = put)

    val dests = cases.map {
        val fnref = newVar().copy(type = it.second.type())
        put(IrInstr(
            mutableListOf(fnref),
            PushFnRefInstr(it.second.uid),
            mutableListOf()
        ))
        fnref
    }.wrapInArgArray(newVar, put = put)

    put(IrInstr(
        dest.toMutableList(),
        PrimitiveInstr(Prims.SWITCH),
        mutableListOf(conds, dests, on).also { it += inputs }
    ))
}

fun filled(
    fillType: Type,
    fillBody: IrBlock.(IrVar) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    putBlock: IrBlock.() -> Unit,
    put: (IrInstr) -> Unit,
    rets: List<IrVar>,
    vararg args: IrVar,
    body: IrBlock.() -> Unit
) {
    val block = IrBlock(anonFnName(), ref).apply {
        fillArg = newVar().copy(type = fillType)

        args.map { newVar().copy(type = it.type).also { this.args += it } }

        rets.map { newVar().copy(type = it.type).also { this.rets += it } }

        body(this)

        putBlock(this)
    }

    val fn = newVar().copy(type = block.type())
    put(IrInstr(
        mutableListOf(fn),
        PushFnRefInstr(block.uid),
        mutableListOf()
    ))

    val fillProv = IrBlock(anonFnName(), ref).apply {
        val out = newVar().copy(type = fillType).also { this.rets += it }

        fillBody.invoke(this, out)

        putBlock(this)
    }

    val fillProvFn = newVar().copy(type = fillProv.type())
    put(IrInstr(
        mutableListOf(fillProvFn),
        PushFnRefInstr(fillProv.uid),
        mutableListOf()
    ))

    put(IrInstr(
        rets.toMutableList(),
        PrimitiveInstr(Prims.FILL),
        mutableListOf(fillProvFn, fn).also { it.addAll(args) },
    ))
}

fun comment(msg: String): IrInstr =
    IrInstr(
        mutableListOf(),
        CommentInstr(msg),
        mutableListOf()
    )

fun unfailure(block: () -> Boolean): Boolean =
    runCatching { block() }.getOrDefault(false)

fun reduceBody(
    put: (IrInstr) -> Unit,
    putBlock: (IrBlock) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    type: Type,
    body: IrBlock.(a: IrVar, b: IrVar, res: IrVar) -> Unit,
): IrVar {
    val reduceBody = IrBlock(anonFnName(), ref).apply {
        val a = newVar().copy(type = type).also { args += it }
        val b = newVar().copy(type = type).also { args += it }

        val res = newVar().copy(type = type).also { rets += it }

        body(a, b, res)

        putBlock(this)
    }

    val reduceBodyFn = newVar().copy(type = reduceBody.type())
    put(IrInstr(
        mutableListOf(reduceBodyFn),
        PushFnRefInstr(reduceBody.uid),
        mutableListOf()
    ))

    return reduceBodyFn
}

fun reduce(
    dest: IrVar,
    arr: IrVar,
    put: (IrInstr) -> Unit,
    putBlock: (IrBlock) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    type: Type,
    body: IrBlock.(a: IrVar, b: IrVar, res: IrVar) -> Unit,
) {
    val reuceBody = reduceBody(put, putBlock, ref, newVar, type, body)

    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prims.REDUCE),
        mutableListOf(reuceBody, arr, reuceBody)
    ))
}

fun reduce(
    arr: IrVar,
    put: (IrInstr) -> Unit,
    putBlock: (IrBlock) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    type: Type,
    body: IrBlock.(a: IrVar, b: IrVar, res: IrVar) -> Unit,
): IrVar {
    val mult = newVar().copy(type = type)
    reduce(mult, arr, put, putBlock, ref, newVar, type, body)
    return mult
}

fun repeat(
    end: IrVar,
    put: (IrInstr) -> Unit,
    putBlock: (IrBlock) -> Unit,
    ref: Map<BlockId, IrBlock>,
    newVar: () -> IrVar,
    args: List<IrVar>,
    body: IrBlock.(idx: IrVar, args: List<IrVar>) -> Unit
) {
    val (zero) = constants(newVar, 0.0, type = Types.size, put = put)

    val block = IrBlock(anonFnName(), ref).apply {
        val idx = this.newVar().copy(type = Types.size).also { this.args += it }
        val args = args.map { this.newVar().copy(type = it.type).also { this.args += it } }

        body(idx, args)

        putBlock(this)
    }

    val fnref = newVar().copy(type = block.type())
    put(IrInstr(
        mutableListOf(fnref),
        PushFnRefInstr(block.uid),
        mutableListOf()
    ))

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prims.Comp.REPEAT),
        mutableListOf(zero, end, fnref).also { it += args }
    ))
}

fun binary(prim: Prim, a: IrVar, b: IrVar, newVar: () -> IrVar, put: (IrInstr) -> Unit): IrVar {
    val t = newVar().copy(type = a.type.combine(b.type))
    put(IrInstr(
        mutableListOf(t),
        PrimitiveInstr(prim),
        mutableListOf(a, b)
    ))
    return t
}

fun decrement(v: IrVar, a: Analysis, newVar: () -> IrVar, put: (IrInstr) -> Unit): IrVar {
    a.constNum(v)?.let {
        return constant(it - 1.0, type = v.type, newVar = newVar, put = put)
    }

    val c1 = constant(1.0, type = v.type, newVar = newVar, put = put)
    return binary(Prims.SUB, c1, v, newVar, put)
}

fun List<IrVar>.reduce(prim: Prim, newVar: () -> IrVar, put: (IrInstr) -> Unit): IrVar =
    reduce { acc, irVar ->
        binary(prim, acc, irVar, newVar, put)
    }

fun ndRepeat(
    sizes: List<IrVar>,
    put: (IrInstr) -> Unit,
    putBlock: (IrBlock) -> Unit,
    ref: Map<String, IrBlock>,
    newVar: () -> IrVar,
    args: List<IrVar>,
    body: IrBlock.(idc: List<IrVar>, args: List<IrVar>) -> Unit
) {
    TODO()
}