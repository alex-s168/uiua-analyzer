package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*

typealias MLIRType = String

object Ty {
    fun memref(shape: List<Int>, type: MLIRType): MLIRType =
        "memref<${shape.map { if (it == -1) "?" else it.toString() }.joinToString(separator = "x")}x$type>"

    fun tensor(shape: List<Int>, type: MLIRType): MLIRType =
        "tensor<${shape.map { if (it == -1) "?" else it.toString() }.joinToString(separator = "x")}x$type>"

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
        is BoxType -> TODO()
        Types.int -> Ty.int(64)
        Types.byte -> Ty.int(8)
        Types.double -> Ty.flt(64)
        Types.dynamic -> TODO()
        is PtrType -> Ty.ptr
        Types.opaque,
        Types.func -> error("should not happen")
        else -> error("")
    }

fun castInstr(from: Type, to: Type, dest: MLIRVar, src: MLIRVar): String =
    when (to) {
        Types.double -> when (from) {
            Types.double -> error("No!")
            is PtrType -> error("double -> ptr not allowed")
            Types.int -> "$dest = arith.sitofp $src : i64 to f64"
            Types.byte -> "$dest = arith.sutofp $src : i8 to f64"
            else -> TODO()
        }
        is PtrType -> when (from) {
            Types.double -> error("ptr -> double not allowed")
            is PtrType -> error("No!")
            Types.int -> "$dest = llvm.inttoptr $src : i64 to !llvm.ptr"
            Types.byte -> "$dest = llvm.inttoptr $src : i8 to !llvm.ptr"
            else -> TODO()
        }
        Types.int -> when (from) {
            Types.double -> "$dest = arith.fptosi $src : f64 to i64"
            is PtrType -> "$dest = llvm.ptrtoint $src : !llvm.ptr to i64"
            Types.int -> error("No!")
            Types.byte -> "$dest = arith.extui $src : i8 to i64"
            else -> TODO()
        }
        Types.byte -> when (from) {
            Types.double -> "$dest = arith.fptoui $src : f64 to i8"
            is PtrType -> "$dest = llvm.ptrtoint $src : !llvm.ptr to i8"
            Types.int -> "$dest = arith.trunci $src : i64 to i8"
            Types.byte -> error("No!")
            else -> TODO()
        }
        else -> TODO()
    }