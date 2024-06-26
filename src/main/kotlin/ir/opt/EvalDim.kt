package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.NumImmInstr
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.optAwayPass
import me.alex_s168.uiua.ir.transform.unfailure

val evalDim = optAwayPass(
    "eval comptime dim",
    { it.instr is PrimitiveInstr && it.instr.id == Prim.Comp.DIM },
    { a -> unfailure {
        val at = args[0].type as ArrayType
        val i = a.origin(args[1])!!.instr as NumImmInstr
        at.shape[i.value.toInt()] != -1
    } }
) { put, newVar, a ->
    val at = args[0].type as ArrayType
    val i = a.origin(args[1])!!.instr as NumImmInstr
    val si = at.shape[i.value.toInt()]

    put(IrInstr(
        mutableListOf(outs[0]),
        NumImmInstr(si.toDouble()),
        mutableListOf()
    ))
}