package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.binary
import me.alex_s168.uiua.ir.transform.constant
import me.alex_s168.uiua.ir.transform.reduce
import me.alex_s168.uiua.ir.transform.wrapInArgArray

val lowerJoin = lowerPrimPass(Prim.JOIN) { put, newVar, a ->
    val lens = args.map {
        val len = newVar().copy(type = Types.size)
        put(IrInstr(
            mutableListOf(len),
            PrimitiveInstr(Prim.LEN),
            mutableListOf(it)
        ))
        len
    }

    val sha = lens.reduce(Prim.ADD, newVar, put)
        .let(::listOf)
        .wrapInArgArray(newVar, put = put)

    val outTy = outs[0].type as ArrayType

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
        mutableListOf(sha)
    ))

    args.zip(lens)
        .fold(constant(0.0, Types.size, newVar, put)) { off, (arr, len) ->
            if (arr.type is ArrayType) {
                val view = newVar().copy(type = outTy.mapShapeElems { -1 })
                put(IrInstr(
                    mutableListOf(view),
                    PrimitiveInstr(Prim.Comp.OFF_VIEW_1D),
                    mutableListOf(outs[0], off, len)
                ))

                put(IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prim.Comp.ARR_COPY),
                    mutableListOf(view, arr)
                ))
            } else {
                put(IrInstr(
                    mutableListOf(),
                    PrimitiveInstr(Prim.Comp.ARR_STORE),
                    mutableListOf(
                        outs[0],
                        listOf(off).wrapInArgArray(newVar, put = put),
                        arr
                    )
                ))
            }

            binary(Prim.ADD, off, len, newVar, put)
        }
}.parallelWithoutDeepCopy()