package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.into

// ⍥⍉-1⧻⊸△
val lowerUnTranspose = lowerPrimPass(Prims.Front.UN_TRANSPOSE) { put, newVar, a ->
    val argTy = args[0].type as ArrayType

    var res = args[0]
    repeat(argTy.shape.size - 1) {
        val new = newVar().copy(type = argTy)
        put(IrInstr(
            mutableListOf(new),
            PrimitiveInstr(Prims.TRANSPOSE),
            mutableListOf(res)
        ))
        res = new
    }

    res.into(outs[0], put)
}