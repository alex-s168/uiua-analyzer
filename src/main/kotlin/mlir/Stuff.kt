package me.alex_s168.uiua.mlir

import blitz.Provider

typealias MLIRVar = String

data class LLVMStruct(
    val types: List<MLIRType>
) {
    val type: MLIRType =
        "!llvm.struct<(${types.joinToString()})>"

    fun setVal(dest: MLIRVar, old: MLIRVar, idx: Int, value: MLIRVar) =
        Inst.insertValue(dest, type, value, old, idx)

    fun getVal(dest: MLIRVar, value: MLIRVar, idx: Int) =
        Inst.extractValue(dest, type, value, idx)

    fun new(newVar: Provider<MLIRVar>, vararg sources: MLIRVar): Pair<List<String>, MLIRVar> {
        val res = mutableListOf<String>()

        val zero = newVar()
        res += Inst.undef(zero, type)

        require(sources.size <= types.size)

        var last = zero
        sources.forEachIndexed { index, s ->
            val va = newVar()
            res += setVal(va, last, index, s)
            last = va
        }

        return res to last
    }
}