package me.alex_s168.uiua.ir.lower

import boundsCheck
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.boundsChecking
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.wrapInArgArray
import me.alex_s168.uiua.ir.withPassArg

val lowerPick = withPassArg<(IrBlock) -> Unit>("lower pick") { putBlock ->
    lowerPrimPass(Prim.PICK) { put, newVar, a ->
        val at = args[0]
        val arr = args[1]

        val idc = listOf(at).wrapInArgArray(newVar, put = put)

        if (boundsChecking)
            boundsCheck(a.block, arr, listOf(idc), newVar, putBlock, put)

        put(IrInstr(
            mutableListOf(outs[0]),
            PrimitiveInstr(Prim.Comp.ARR_LOAD),
            mutableListOf(arr, idc)
        ))
    }
}