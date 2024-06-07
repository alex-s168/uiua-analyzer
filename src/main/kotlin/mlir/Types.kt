package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*

typealias MLIRType = String

fun List<Int>.shapeToMLIR(): String =
    map { if (it == -1) "?" else it.toString() }.joinToString(separator = "x")

fun List<Int>.shapeToMLIRStrides(): String =
    drop(1).plus(1).shapeToMLIR()

object Ty {
    fun memref(shape: List<Int>, type: MLIRType): MLIRType =
        "memref<${shape.shapeToMLIR()} x $type, strided<[${shape.shapeToMLIRStrides()}]>>"

    fun tensor(shape: List<Int>, type: MLIRType): MLIRType =
        "tensor<${shape.shapeToMLIR()} x $type, strided<[${shape.shapeToMLIRStrides()}]>>"

    // https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
    fun memrefStruct(shape: List<Int>) =
        LLVMStruct(listOf(ptr, ptr, index, array(shape.size, index), array(shape.size, index)))

    fun array(size: Int, type: MLIRType) =
        "array<$size x $type>"

    fun int(size: Int): MLIRType =
        "i$size"

    fun flt(size: Int): MLIRType =
        "f$size"

    val index = "index"

    val ptr = "!llvm.ptr"
}

fun ptrlit(literal: String) =
    "!llvm.ptr(<$literal>)"

fun Type.toMLIR(wantTensor: Boolean = false): MLIRType =
    when (this) {
        is ArrayType -> inner.toMLIR(wantTensor).let {
            if (wantTensor) Ty.tensor(shape, it)
            else Ty.memref(shape, it)
        }
        is BoxType -> Ty.memref(listOf(1), of.toMLIR(wantTensor))
        Types.int -> Ty.int(64)
        Types.byte -> Ty.int(8)
        Types.double -> Ty.flt(64)
        Types.dynamic -> TODO()
        is PtrType -> Ty.ptr
        Types.opaque -> error("should not happen")
        is FnType -> "(${(fillType?.let { listOf(it) + args } ?: args).joinToString { it.toMLIR(wantTensor) }}) -> (${rets.joinToString { it.toMLIR(wantTensor) }})"
        else -> error("")
    }

fun castInstr(from: Type, to: Type, dest: MLIRVar, src: MLIRVar): String =
    when (to) {
        Types.double -> when (from) {
            Types.double -> error("No!")
            is PtrType -> error("double -> ptr not allowed")
            Types.int -> "$dest = arith.sitofp $src : i64 to f64"
            Types.byte -> "$dest = arith.sutofp $src : i8 to f64"
            else -> error("Cast from $from to $to not implemented")
        }
        is PtrType -> when (from) {
            Types.double -> error("ptr -> double not allowed")
            is PtrType -> error("No!")
            Types.int -> "$dest = llvm.inttoptr $src : i64 to !llvm.ptr"
            Types.byte -> "$dest = llvm.inttoptr $src : i8 to !llvm.ptr"
            else -> error("Cast from $from to $to not implemented")
        }
        Types.int -> when (from) {
            Types.double -> "$dest = arith.fptosi $src : f64 to i64"
            is PtrType -> "$dest = llvm.ptrtoint $src : !llvm.ptr to i64"
            Types.int -> error("No!")
            Types.byte -> "$dest = arith.extui $src : i8 to i64"
            else -> error("Cast from $from to $to not implemented")
        }
        Types.byte -> when (from) {
            Types.double -> "$dest = arith.fptoui $src : f64 to i8"
            is PtrType -> "$dest = llvm.ptrtoint $src : !llvm.ptr to i8"
            Types.int -> "$dest = arith.trunci $src : i64 to i8"
            Types.byte -> error("No!")
            else -> error("Cast from $from to $to not implemented")
        }
        else -> error("Cast from $from to $to not implemented")
    }