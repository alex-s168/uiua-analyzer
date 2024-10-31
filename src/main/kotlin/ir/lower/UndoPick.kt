package me.alex_s168.uiua.ir.lower

import boundsCheck
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.boundsChecking
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerUndoPick = withPassArg<(IrBlock) -> Unit>("lower unpick") { putBlock ->
    lowerPrimPass(Prims.UNDO_PICK) { put, newVar, a ->
        val at = args[1]
        val arr = args[0]
        val value = args[2]

        put(IrInstr(
            mutableListOf(outs[0]),
            PrimitiveInstr(Prims.Comp.ARR_CLONE),
            mutableListOf(arr)
        ))

        val idc = listOf(at).wrapInArgArray(newVar, put = put)

        if (boundsChecking)
            boundsCheck(a.block, arr, listOf(at), newVar, putBlock, put)

        put(IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prims.Comp.ARR_STORE),
            mutableListOf(outs[0], idc, value)
        ))
    }
}.parallelWithoutDeepCopy()