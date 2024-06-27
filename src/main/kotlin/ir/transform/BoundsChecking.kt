package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.*

val boundsChecking = withPassArg<(IrBlock) -> Unit>("bounds checking") { putBlock ->
    modifyPass(
        "bounds checking",
        { (it.instr is PrimitiveInstr) && (it.instr.id in Analysis.boundChecked) },
        { a -> true },
    ) { put, newVar, a ->
        // this is put in last

        instr as PrimitiveInstr

        val (arr, idxList) = Analysis.boundChecked[instr.id]!!.toList()
            .map { args[it] }

        val arrTy = arr.type as ArrayType

        val indexes = a.origin(idxList)!!.args

        indexes.forEachIndexed { shaIdx, index ->
            val (shaIdxV) = constants(newVar, shaIdx.toDouble(), type = Types.size, put = put)

            val atShape = arrTy.shape[shaIdx]

            if (atShape != -1) {
                a.origin(index)?.instr?.let { idxDecl ->
                    if (idxDecl is NumImmInstr) {
                        val idx = idxDecl.value.toInt()
                        if (idx >= atShape) {
                            error("(comptime catched) Index $idx out of bounds for array with size $atShape (in shape)")
                        }
                        return@forEachIndexed
                    }
                }
            }

            val dim = newVar().copy(type = Types.size)
            put(IrInstr(
                mutableListOf(dim),
                PrimitiveInstr(Prim.Comp.DIM),
                mutableListOf(arr, shaIdxV)
            ))

            val lt = newVar().copy(type = Types.bool)
            put(IrInstr(
                mutableListOf(lt),
                PrimitiveInstr(Prim.LT),
                mutableListOf(dim, index) // index < dim
            ))

            val nop = IrBlock(anonFnName(), a.block.ref).apply {
                putBlock(this)
            }

            val panic = IrBlock(anonFnName(), a.block.ref).apply {
                instrs += IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prim.Comp.PANIC),
                    mutableListOf()
                )

                putBlock(this)
            }

            val (zero, one) = constants(newVar, 0.0, 1.0, type = Types.bool, put = put)

            switch(
                listOf(),
                newVar,
                lt,
                listOf(),
                one to nop, // index < dim
                zero to panic, // index >= dim
                put = put,
            )
        }
    }
}