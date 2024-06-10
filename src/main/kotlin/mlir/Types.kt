package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrVar

typealias MLIRType = String

fun List<Int>.shapeToMLIR() =
    map { if (it == -1) "?" else it.toString() }

fun List<Int>.shapeToMLIRStrides(): String =
    drop(1).plus(1).shapeToMLIR().joinToString(separator = "x")

object Ty {
    fun memref(shape: List<Int>, type: MLIRType): MLIRType =
        "memref<${shape.shapeToMLIR().joinToString(separator = "x")} x $type, strided<[${shape.shapeToMLIRStrides()}]>>"

    fun tensor(shape: List<Int>, type: MLIRType): MLIRType =
        "tensor<${shape.shapeToMLIR().joinToString(separator = "x")} x $type, strided<[${shape.shapeToMLIRStrides()}]>>"

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
        Types.bool -> Ty.int(1)
        Types.double -> Ty.flt(64)
        Types.dynamic -> TODO()
        is PtrType -> Ty.ptr
        Types.opaque -> error("should not happen")
        is FnType -> "(${(fillType?.let { listOf(it) + args } ?: args).joinToString { it.toMLIR(wantTensor) }}) -> (${rets.joinToString { it.toMLIR(wantTensor) }})"
        Types.size -> Ty.index
        else -> error("$this")
    }

fun castInstr(newVar: () -> IrVar, from: Type, to: Type, dest: MLIRVar, src: MLIRVar): List<String> =
    when (to) {
        Types.double -> when (from) {
            Types.double -> error("No!")
            is PtrType -> error("double -> ptr not allowed")
            Types.int -> listOf("$dest = arith.sitofp $src : i64 to f64")
            Types.size -> {
                val v = newVar().copy(type = Types.int).asMLIR()
                listOf(
                    "$v = index.castu $src : index to i64",
                    "$dest = arith.uitofp $v : i64 to f64"
                )
            }
            Types.bool -> listOf("$dest = arith.uitofp $src : i1 to f64")
            Types.byte -> listOf("$dest = arith.uitofp $src : i8 to f64")
            else -> error("Cast from $from to $to not implemented")
        }
        is PtrType -> when (from) {
            Types.double -> error("ptr -> double not allowed")
            is PtrType -> error("No!")
            Types.int -> listOf("$dest = llvm.inttoptr $src : i64 to !llvm.ptr")
            Types.size -> listOf("$dest = llvm.inttoptr $src : index to !llvm.ptr")
            Types.byte -> listOf("$dest = llvm.inttoptr $src : i8 to !llvm.ptr")
            Types.bool -> listOf("$dest = llvm.inttoptr $src : i1 to !llvm.ptr")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.int -> when (from) {
            Types.double -> listOf("$dest = arith.fptosi $src : f64 to i64")
            is PtrType -> listOf("$dest = llvm.ptrtoint $src : !llvm.ptr to i64")
            Types.int -> error("No!")
            Types.size -> listOf("$dest = index.castu $src : index to i64")
            Types.byte -> listOf("$dest = arith.extui $src : i8 to i64")
            Types.bool -> listOf("$dest = arith.extui $src : i1 to i64")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.byte -> when (from) {
            Types.double -> listOf("$dest = arith.fptoui $src : f64 to i8")
            is PtrType -> listOf("$dest = llvm.ptrtoint $src : !llvm.ptr to i8")
            Types.int -> listOf("$dest = arith.trunci $src : i64 to i8")
            Types.size -> listOf("$dest = index.castu $src : index to i8")
            Types.bool -> listOf("$dest = arith.extui $src : i1 to i8")
            Types.byte -> error("No!")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.bool -> when (from) {
            Types.double -> listOf("$dest = arith.fptoui $src : f64 to i1")
            is PtrType -> listOf("$dest = llvm.ptrtoint $src : !llvm.ptr to i1")
            Types.int -> listOf("$dest = arith.trunci $src : i64 to i1")
            Types.size -> listOf("$dest = index.castu $src : index to i1")
            Types.byte -> listOf("$dest = arith.trunci $src : i64 to i1")
            Types.bool -> error("no")
            else -> error("Cast from $from to $to not implemented")
        }
        Types.size -> when (from) {
            Types.double -> {
                val v = newVar().copy(type = Types.int).asMLIR()
                listOf(
                    "$v = arith.fptoui $src : f64 to i64",
                    "$dest = index.castu $v : i64 to index"
                )
            }
            Types.int -> listOf("$dest = index.castu $src : i64 to index")
            Types.byte -> listOf("$dest = index.castu $src : i8 to index")
            Types.bool -> listOf("$dest = index.castu $src : i1 to index")
            is PtrType -> listOf("$dest = llvm.ptrtoint $src : !llvm.ptr to index")
            Types.size -> error("No!")
            else -> error("Cast from $from to $to not implemented")
        }
        else -> error("Cast from $from to $to not implemented")
    }