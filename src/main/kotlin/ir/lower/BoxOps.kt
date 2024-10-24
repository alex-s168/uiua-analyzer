package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrInstr
import me.alex_s168.uiua.ir.transform.constantArr
import me.alex_s168.uiua.ir.lowerPrimPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy
import me.alex_s168.uiua.ir.transform.oneDimLoad

val lowerBoxCreate = lowerPrimPass(Prim.Comp.BOX_CREATE) { put, newVar, a ->
    val shape = constantArr(newVar, 1.0, type = Types.int, put = put)

    put(IrInstr(
        mutableListOf(outs[0]),
        PrimitiveInstr(Prim.Comp.ARR_ALLOC),
        mutableListOf(shape)
    ))
}.parallelWithoutDeepCopy()

val lowerBoxDestroy = lowerPrimPass(Prim.Comp.BOX_DESTROY) { put, newVar, a ->
    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.ARR_DESTROY),
        mutableListOf(args[0])
    ))
}.parallelWithoutDeepCopy()

val lowerBoxLoad = lowerPrimPass(Prim.Comp.BOX_LOAD) { put, newVar, a ->
    oneDimLoad(dest = outs[0], args[0], newVar, 0, put = put)
}.parallelWithoutDeepCopy()

val lowerBoxStore = lowerPrimPass(Prim.Comp.BOX_STORE) { put, newVar, a ->
    val indecies = constantArr(newVar, 0.0, type = Types.int, put = put)

    put(IrInstr(
        mutableListOf(),
        PrimitiveInstr(Prim.Comp.ARR_STORE),
        mutableListOf(args[0], indecies, args[1])
    ))
}.parallelWithoutDeepCopy()