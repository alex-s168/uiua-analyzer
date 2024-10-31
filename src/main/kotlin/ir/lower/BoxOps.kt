package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constantArr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.oneDimLoad

val lowerBoxCreate = lowerPrimPass(Prims.Comp.BOX_CREATE) { put, newVar, a ->
    val shape = constantArr(newVar, 1.0, type = Types.int, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prims.Comp.ARR_ALLOC),
        mutableListOf(shape)
    ))
}.parallelWithoutDeepCopy()

val lowerBoxDestroy = lowerPrimPass(Prims.Comp.BOX_DESTROY) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prims.Comp.ARR_DESTROY),
        mutableListOf(args[0])
    ))
}.parallelWithoutDeepCopy()

val lowerBoxLoad = lowerPrimPass(Prims.Comp.BOX_LOAD) { put, newVar, a ->
    oneDimLoad(dest = outs[0], args[0], newVar, 0, put = put)
}.parallelWithoutDeepCopy()

val lowerBoxStore = lowerPrimPass(Prims.Comp.BOX_STORE) { put, newVar, a ->
    val indecies = constantArr(newVar, 0.0, type = Types.int, put = put)

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prims.Comp.ARR_STORE),
        mutableListOf(args[0], indecies, args[1])
    ))
}.parallelWithoutDeepCopy()