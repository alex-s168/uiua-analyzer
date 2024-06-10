package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun List<IrVar>.wrapInArgArray(newVar: () -> IrVar, type: Type = first().type, put: (IrInstr) -> Unit): IrVar {
    val v = newVar().copy(type = Types.array(type))
    put(IrInstr(
        mutableListOf(v),
        PrimitiveInstr(Prim.Comp.ARG_ARR),
        toMutableList()
    ))
    return v
}

fun IrVar.into(dest: IrVar, put: (IrInstr) -> Unit) {
    if (this.id != dest.id) {
        put(IrInstr(
            mutableListOf(dest),
            PrimitiveInstr(Prim.Comp.USE),
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

fun constantArr(newVar: () -> IrVar, vararg const: Double, type: Type = Types.double, put: (IrInstr) -> Unit): IrVar =
    constants(newVar, *const, type = type, put = put).wrapInArgArray(newVar, type, put)

fun constantArr(dest: IrVar, newVar: () -> IrVar, vararg const: Double, type: Type = Types.double, put: (IrInstr) -> Unit) {
    constantArr(newVar, *const, type = type, put = put).into(dest, put)
}

fun oneDimLoad(dest: IrVar, arr: IrVar, newVar: () -> IrVar, idx: Int, put: (IrInstr) -> Unit) {
    val indecies = constantArr(newVar, idx.toDouble(), type = Types.int, put = put)

    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prim.Comp.ARR_LOAD),
        mutableListOf(arr, indecies)
    ))
}

fun oneDimLoad(arr: IrVar, newVar: () -> IrVar, idx: Int, put: (IrInstr) -> Unit): IrVar {
    val variable = newVar().copy(type = (arr.type as ArrayType).of)
    oneDimLoad(dest = variable, arr = arr, newVar = newVar, idx = idx, put = put)
    return variable
}

fun oneDimLoad(dest: IrVar, arr: IrVar, newVar: () -> IrVar, idx: IrVar, put: (IrInstr) -> Unit) {
    val indecies = listOf(idx).wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prim.Comp.ARR_LOAD),
        mutableListOf(arr, indecies)
    ))
}

fun oneDimLoad(arr: IrVar, newVar: () -> IrVar, idx: IrVar, put: (IrInstr) -> Unit): IrVar {
    val variable = newVar().copy(type = (arr.type as ArrayType).of)
    oneDimLoad(dest = variable, arr = arr, newVar = newVar, idx = idx, put = put)
    return variable
}

fun oneDimFillLoad(dest: IrVar, fill: IrVar?, arr: IrVar, putBlock: (IrBlock) -> Unit, ref: (String) -> IrBlock?, newVar: () -> IrVar, idx: IrVar, put: (IrInstr) -> Unit) {
    arr.type as ArrayType

    val err = IrBlock(anonFnName(), ref).apply {
        fillArg = fill?.let { newVar().copy(type = it.type) }

        val arr = newVar().copy(type = arr.type).also { args += it }
        val idx = newVar().copy(type = idx.type).also { args += it }

        val value = newVar().copy(type = arr.type).also { rets += it }

        if (fillArg == null) {
            instrs += IrInstr(
                mutableListOf(value),
                PrimitiveInstr(Prim.Comp.PANIC),
                mutableListOf()
            )
        } else {
            fillArg!!.into(value) { instrs += it }
        }

        putBlock(this)
    }

    val ok = IrBlock(anonFnName(), ref).apply {
        val arr = newVar().copy(type = arr.type).also { args += it }
        val idx = newVar().copy(type = idx.type).also { args += it }

        val value = newVar().copy(type = arr.type).also { rets += it }

        val idc = listOf(idx).wrapInArgArray(::newVar) { instrs += it }

        instrs += IrInstr(
            mutableListOf(value),
            PrimitiveInstr(Prim.Comp.ARR_LOAD),
            mutableListOf(arr, idc)
        )

        putBlock(this)
    }

    val len = newVar().copy(type = Types.size)
    put(IrInstr(
        mutableListOf(len),
        PrimitiveInstr(Prim.LEN),
        mutableListOf(arr)
    ))

    val lt = newVar().copy(type = Types.byte)
    put(IrInstr(
        mutableListOf(lt),
        PrimitiveInstr(Prim.LT),
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

fun oneDimFillLoad(fill: IrVar?, arr: IrVar, putBlock: (IrBlock) -> Unit, ref: (String) -> IrBlock?, newVar: () -> IrVar, idx: IrVar, put: (IrInstr) -> Unit): IrVar {
    val variable = newVar().copy(type = (arr.type as ArrayType).of)
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
            PushFnRefInstr(it.second.name),
            mutableListOf()
        ))
        fnref
    }.wrapInArgArray(newVar, put = put)

    put(IrInstr(
        dest.toMutableList(),
        PrimitiveInstr(Prim.SWITCH),
        mutableListOf(conds, dests, on).also { it += inputs }
    ))
}

fun filled(
    fillType: Type,
    fillBody: IrBlock.(IrVar) -> Unit,
    ref: (String) -> IrBlock?,
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
        PushFnRefInstr(block.name),
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
        PushFnRefInstr(fillProv.name),
        mutableListOf()
    ))

    put(IrInstr(
        rets.toMutableList(),
        PrimitiveInstr(Prim.FILL),
        mutableListOf(fillProvFn, fn).also { it.addAll(args) },
    ))
}

fun comment(msg: String): IrInstr =
    IrInstr(
        mutableListOf(),
        CommentInstr(msg),
        mutableListOf()
    )