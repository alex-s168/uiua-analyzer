package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.ir.modifyPass


// TODO: THIS IS EXTREMLY UNSAFE
val loadStore = modifyPass(
    "load store",
    Prims.Comp.ARR_LOAD,
    {true}
) { put, newVar, a ->
    val arr = args[0]

    val stores = a.usages(arr)
        .filterNotNull()
        .filter { a.isPrim(it, Prims.Comp.ARR_STORE) }

    val sto = stores.firstOrNull {
        it.args[0] == arr &&
            a.identical(it.args[1], args[1])
    }

    sto?.let {
        println("yay")
        args.clear()
        (instr as PrimitiveInstr).id = Prims.Comp.USE
        args[0] = sto.args[2]
    }
}