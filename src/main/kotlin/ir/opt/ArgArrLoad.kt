package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.ir.modifyPass

val argArrLoad = modifyPass(
    "const fold arg arr loads",
    Prims.Comp.ARR_LOAD,
    {true}
) { put, newVar, a ->
    val argArr =
        a.deepOriginV2(a.argArrVar(args[0]))?.a?.second
            ?: return@modifyPass
    if (!a.isPrim(argArr, Prims.Comp.ARG_ARR))
        return@modifyPass
    val idx = a.argArrAsConsts(args[1])
        ?: return@modifyPass
    if (idx.size != 1)
        return@modifyPass

    (instr as PrimitiveInstr).id = Prims.Comp.USE
    args[0] = argArr.args[idx[0].toInt()]
    args.removeAt(1)
}