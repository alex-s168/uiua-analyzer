package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.constant
import me.alex_s168.uiua.ir.transform.reduce
import me.alex_s168.uiua.ir.transform.wrapInArgArray
import me.alex_s168.uiua.ir.transform.repeat

val lowerArrCopy = lowerPrimPass<(IrBlock) ->  Unit>(Prims.Comp.ARR_COPY) { put, newVar, a, putBlock ->
    val dest = args[0]
    val src = args[1]

    val arrTy = src.type as ArrayType
    val destTy = dest.type as ArrayType

    val shape = List(arrTy.shape.size) {
        val idx = constant(it.toDouble(), Types.size, newVar, put)
        val len = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(len),
            PrimitiveInstr(Prims.Comp.DIM),
            mutableListOf(src, idx)
        ))
        len
    }

    val numTotal = shape.reduce(Prims.MUL, newVar, put)
    val desShape = listOf(numTotal).wrapInArgArray(newVar, put=put)

    val (srcDes, destDes) = if (destTy.shape.size != 1) {
        val srcDes = newVar().copy(type = Types.array(arrTy.inner))
        put(IrInstr(
            mutableListOf(srcDes),
            PrimitiveInstr(Prims.Comp.RESHAPE_VIEW),
            mutableListOf(desShape, src)
        ))

        val destDes = newVar().copy(type = Types.array(destTy.inner))
        put(IrInstr(
            mutableListOf(destDes),
            PrimitiveInstr(Prims.Comp.RESHAPE_VIEW),
            mutableListOf(desShape, dest)
        ))

        srcDes to destDes
    } else src to dest

    repeat(numTotal, put, putBlock, a.block.ref, newVar, listOf(destDes, srcDes)) { idx, (destDes, srcDes) ->
        val idxa = listOf(idx).wrapInArgArray(newVar, put = instrs::add)
        val temp = newVar().copy(type = arrTy.inner)
        instrs.add(IrInstr(
            mutableListOf(temp),
            PrimitiveInstr(Prims.Comp.ARR_LOAD),
            mutableListOf(srcDes, idxa)
        ))
        instrs.add(IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prims.Comp.ARR_STORE),
            mutableListOf(destDes, idxa, temp)
        ))
    }
}