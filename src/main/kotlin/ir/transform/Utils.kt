package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar

fun List<IrVar>.wrapInArgArray(newVar: () -> IrVar, put: (IrInstr) -> Unit): IrVar {
    val v = newVar().copy(type = Types.array(first().type))
    put(IrInstr(
        mutableListOf(v),
        PrimitiveInstr(Prim.Comp.ARG_ARR),
        toMutableList()
    ))
    return v
}

fun IrVar.into(dest: IrVar, put: (IrInstr) -> Unit) {
    put(IrInstr(
        mutableListOf(dest),
        PrimitiveInstr(Prim.Comp.USE),
        mutableListOf(this)
    ))
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
    constants(newVar, *const, type = type, put = put).wrapInArgArray(newVar, put)

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
    val indecies = listOf(idx).wrapInArgArray(newVar, put)

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

fun switch(
    dest: List<IrVar>,
    newVar: () -> IrVar,
    on: IrVar,
    inputs: List<IrVar>,
    vararg cases: Pair<IrVar, IrBlock>,
    put: (IrInstr) -> Unit
) {
    val conds = cases.map { it.first }
        .wrapInArgArray(newVar, put)

    val dests = cases.map {
        val fnref = newVar().copy(type = it.second.type())
        put(IrInstr(
            mutableListOf(fnref),
            PushFnRefInstr(it.second.name),
            mutableListOf()
        ))
        fnref
    }.wrapInArgArray(newVar, put)

    put(IrInstr(
        dest.toMutableList(),
        PrimitiveInstr(Prim.SWITCH),
        mutableListOf(conds, dests, on).also { it += inputs }
    ))
}