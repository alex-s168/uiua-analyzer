package me.alex_s168.uiua.ir.lower

import blitz.flatten
import blitz.mapA
import me.alex_s168.uiua.ArrImmInstr
import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prims
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

    constantArr(outs[0], newVar, *values.toDoubleArray(), put = put)
}.parallelWithoutDeepCopy()
