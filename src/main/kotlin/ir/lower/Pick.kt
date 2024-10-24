package me.alex_s168.uiua.ir.lower

import boundsCheck
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.boundsChecking
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerPick = withPassArg<(IrBlock) -> Unit>("lower pick") { putBlock ->
    lowerPrimPass(Prim.PICK) { put, newVar, a ->
        val at = args[0]
        val arr = args[1]

        val idc = listOf(at).wrapInArgArray(newVar, put = put)

        if (boundsChecking)
            boundsCheck(a.block, arr, listOf(at), newVar, putBlock, put)

        put(IrInstr(
            mutableListOf(outs[0]),
            PrimitiveInstr(Prim.Comp.ARR_LOAD),
            mutableListOf(arr, idc)
        ))
    }
}.parallelWithoutDeepCopy()