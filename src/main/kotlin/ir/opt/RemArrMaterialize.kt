package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prim
import me.alex_s168.uiua.ir.Analysis
import me.alex_s168.uiua.ir.allPrim
import me.alex_s168.uiua.ir.optAwayPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

var remArrMat = optAwayPass(
    "opt away array materialize",
    Prim.Comp.ARR_MATERIALIZE,
    { a ->
        a.usages(outs[0]).allPrim { it in Analysis.argArrayUsing.keys }
    },
    { put, newVar, a ->
        a.rename(outs[0], args[0])
    }
).parallelWithoutDeepCopy()