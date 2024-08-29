package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.shapeToArrType

fun subview(newVar: () -> IrVar, body: MutableList<String>, dest: IrVar, arr: IrVar, indecies: List<IrVar>) {
    val arrTy = arr.type as ArrayType
    val dims = List(arrTy.shape.size) { i ->
        val const = newVar().copy(type = Types.size)
        body += Inst.constant(const.asMLIR(), const.type.toMLIR(), "$i")
        val v = newVar().copy(type = Types.size)
        body += Inst.memRefDim(v.asMLIR(), arrTy.toMLIR(), arr.asMLIR(), const.asMLIR())
        v
    }
    /*
    val offsets = indecies.map { it.asMLIR() } + List(arrTy.shape.size) { "0" }
    val size = arrTy.shape.mapIndexed { i, s -> if (i < indecies.size) 1 else s }.shapeToMLIR()
    val strides = arrTy.shape.map { "1" }
     */
    val offsets = indecies.map { castIfNec(newVar, body, it, Types.size).asMLIR() } + List(arrTy.shape.size - indecies.size) { 0 }
    val size = indecies.map { 1 }.shapeToMLIR() + List(arrTy.shape.size - indecies.size) { dims[it + indecies.size].asMLIR() }
    val strides = dims.map { it.asMLIR() }.drop(1).plus("1")

    body += "${dest.asMLIR()} = memref.subview ${arr.asMLIR()}[${offsets.joinToString()}][${size.joinToString()}][${strides.joinToString()}] : \n  ${arr.type.toMLIR()} to ${dest.type.toMLIR()}"
}

fun subview(newVar: () -> IrVar, body: MutableList<String>, arr: IrVar, indecies: List<IrVar>): IrVar {
    val arrTy = arr.type as ArrayType
    val dest = newVar().copy(type = arrTy.shape.drop(indecies.size).shapeToArrType(arrTy.inner).copy(vaOff = true))
    subview(newVar, body, dest, arr, indecies)
    return dest
}
