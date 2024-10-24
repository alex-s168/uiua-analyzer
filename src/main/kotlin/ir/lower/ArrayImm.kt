package me.alex_s168.uiua.ir.lower

import blitz.flatten
import me.alex_s168.uiua.ArrImmInstr
import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constantArr
import me.alex_s168.uiua.ir.lowerPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val lowerArrImm = lowerPass("lower arr imm", { it.instr is ArrImmInstr }) { put, newVar, a ->
    instr as ArrImmInstr

    val values = instr.values
        .mapA { it.map { it.toDouble() } }
        .flatten()

    val arrType = outs[0].type as ArrayType

    val arr = newVar().copy(type = arrType)
    constantArr(arr, newVar, *values.toDoubleArray(), put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_MATERIALIZE),
        mutableListOf(arr)
    ))
}.parallelWithoutDeepCopy()