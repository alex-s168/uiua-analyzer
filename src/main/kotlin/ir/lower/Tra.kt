package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.BlockId
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.transform.constants
import me.alex_s168.uiua.ir.transform.wrapInArgArray
import me.alex_s168.uiua.ir.transform.ndRepeat

val lowerTranspose = lowerPrimPass<(IrBlock) -> Unit>(Prims.TRANSPOSE) { put, newVar, a, putBlock ->
    val arrTy = args[0].type as ArrayType

    // TODO: move this to utily functon
    val shaElems = List(arrTy.shape.size) {
        val (d) = constants(newVar, it.toDouble(), type = Types.size, put = put)

        val dim = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(dim),
            PrimitiveInstr(Prims.Comp.DIM),
            mutableListOf(args[0], d)
        ))
        dim
    }

    val traSha = shaElems.reversed().wrapInArgArray(newVar, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.ARR_ALLOC),
        mutableListOf(traSha)
    ))

    // doing transpose manually since mlir hates me

    ndRepeat(
        shaElems,
        put,
        putBlock,
        a.block.ref,
        newVar,
        listOf(outs[0], args[0])
    ) { idc, args ->
        val outp = args[0]
        val inp = args[1]

        val idca = idc.wrapInArgArray(this::newVar, put = this.instrs::add)
        val revidca = idc.reversed().wrapInArgArray(this::newVar, put = this.instrs::add)

        val value = this.newVar().copy(type = arrTy.inner)
        this.instrs += IrInstr(
            mutableListOf(value),
            PrimitiveInstr(Prims.Comp.ARR_LOAD),
            mutableListOf(inp, idca)
        )
        this.instrs += IrInstr(
            mutableListOf(),
            PrimitiveInstr(Prims.Comp.ARR_STORE),
            mutableListOf(outp, revidca, value)
        )
    }

    // TODO: FIX MLIR TRANSPOSE CODEGEN
    /*put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prims.Comp.TRANSPOSE),
        mutableListOf(outs[0], args[0])
    ))*/
}
