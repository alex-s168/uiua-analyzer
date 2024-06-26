package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerUndoPick = lowerPrimPass(Prim.UNDO_PICK) { put, newVar, a ->
    val at = args[1]
    val arr = args[0]
    val value = args[2]

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_CLONE),
        mutableListOf(arr)
    ))

    val idc = listOf(at).wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.ARR_STORE),
        mutableListOf(outs[0], idc, value)
    ))
}