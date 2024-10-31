package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.ir.optAwayPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val inlineCUse = optAwayPass(
    "inline cUSE",
    Prims.Comp.USE,
    { true },
    { put, newVar, a ->
        a.rename(outs[0], args[0])
    }
).parallelWithoutDeepCopy()