package me.alex_s168.uiua.mlir

import me.alex_s168.uiua.*

class NewlyAllocArrayType(
    of: Type,
    len: Int?
): ArrayType(of, len, false) {
    companion object {
        fun from(ty: ArrayType) =
            NewlyAllocArrayType(ty.of, ty.length)
    }
}

typealias MLIRType = String

fun List<Int>.shapeToMLIR() =
    map { if (it == -1) "?" else it.toString() }

fun List<Int>.shapeToMLIRStrides(): String =
    drop(1).plus(1).shapeToMLIR().joinToString(separator = ",")

fun List<Int>.shapeToMLIRStrided(vaOff: Boolean) =
    "strided<[${shapeToMLIRStrides()}]${if (vaOff) ", offset: ?" else ""}>"

object Ty {
    fun memref(shape: List<Int>, type: MLIRType, vaOff: Boolean, extra: String = ""): MLIRType =
        "memref<${shape.shapeToMLIR().joinToString(separator = "x")} x $type, ${shape.shapeToMLIRStrided(vaOff)}$extra>"

    fun tensor(shape: List<Int>, type: MLIRType, vaOff: Boolean, extra: String = ""): MLIRType =
        "tensor<${shape.shapeToMLIR().joinToString(separator = "x")} x $type, ${shape.shapeToMLIRStrided(vaOff)}$extra>"

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

    val dyn = LLVMStruct(listOf(int(8), ptr))
}

fun ptrlit(literal: String) =
    "!llvm.ptr(<$literal>)"

fun Type.toMLIR(wantTensor: Boolean = false): MLIRType =
    when (this) {
        is NewlyAllocArrayType -> "memref<${shape.shapeToMLIR().joinToString(separator = "x")} x ${inner.toMLIR(wantTensor)}>"
        is ArrayType -> inner.toMLIR(wantTensor).let {
            if (wantTensor) Ty.tensor(shape, it, vaOff)
            else Ty.memref(shape, it, vaOff)
        }
        is BoxType -> Ty.memref(listOf(1), of.toMLIR(wantTensor), false)
        Types.int -> Ty.int(64)
        Types.byte -> Ty.int(8)
        Types.bool -> Ty.int(1)
        Types.double -> Ty.flt(64)
        Types.dynamic -> Ty.dyn.type
        is PtrType -> Ty.ptr
        Types.opaque -> error("should not happen")
        is FnType -> "(${(fillType?.let { listOf(it) + args } ?: args).joinToString { it.toMLIR(wantTensor) }}) -> (${rets.joinToString { it.toMLIR(wantTensor) }})"
        Types.size -> Ty.index
        Types.complex -> "complex<${Ty.flt(64)}>"
        else -> error("$this")
    }

fun Type.cName(): String =
    when (this) {
        Types.double -> "double"
        Types.autobyte,
        Types.byte -> "uint8_t"
        Types.size -> "size_t"
        Types.bool -> "bool"
        Types.dynamic -> "uac_Dyn"
        Types.int -> "int64_t"
        Types.complex -> "uac_Complex"

        is ArrayType -> "Arru_${inner.cName()}"

        else -> error("invalid")
    }
