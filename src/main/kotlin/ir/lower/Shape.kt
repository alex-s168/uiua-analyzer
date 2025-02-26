package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.wrapInArgArray
import me.alex_s168.uiua.ir.transform.into

val lowerShape = lowerPrimPass(Prims.SHAPE) { put, newVar, a ->
    val arr = args[0]
    val arrTy = arr.type as ArrayType

    List(arrTy.shape.size) { dim ->
        val const = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(const),
            NumImmInstr(dim.toDouble()),
            mutableListOf()
        ))

        val v = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(v),
            PrimitiveInstr(Prims.Comp.DIM),
            mutableListOf(arr, const)
        ))

        v
    }.wrapInArgArray(newVar, put = put).into(outs[0], put)
}.parallelWithoutDeepCopy()
