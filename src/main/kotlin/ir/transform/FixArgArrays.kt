package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.ArrayType
import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.merge
import me.alex_s168.uiua.ir.modifyPass

private fun genFor(prim: Prim, argIdx: Int) =
    modifyPass(
        "arg arr fix for ${Prims.all[prim]}",
        prim,
        { a ->
            a.origin(args[argIdx])?.let { !a.isPrim(it, Prims.Comp.ARG_ARR) } ?: false
        },
        { put, newVar, a ->
            val sha = a.argArrVar(args[argIdx])

            require(sha.type is ArrayType)
            require(sha.type.length != null)

            val elem = List(sha.type.length) { x ->
                oneDimLoad(sha, newVar, x, put)
            }.wrapInArgArray(newVar, Types.size, put)

            args[argIdx] = elem
        }
    )

val fixArgArrays = Analysis.argArrayUsing
    .map { (k, v) -> genFor(k, v) }
    .merge("fix arg arrays")