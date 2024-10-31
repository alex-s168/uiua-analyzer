package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.Prims
import me.alex_s168.uiua.PrimitiveInstr
import me.alex_s168.uiua.ir.Lifetime
import me.alex_s168.uiua.ir.modifyPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val remClone = modifyPass(
    "optimize away clone",
    { it.instr is PrimitiveInstr && it.instr.id == Prims.Comp.ARR_CLONE },
    { a ->
        val array = args[0]
        val laterUsages = a.usagesAfter(array, this)
        val lifetime = a.block.lifetimes[array]!!
        laterUsages.none() && lifetime == Lifetime.LOCAL
    },
    { put, newVar, a ->
        instr as PrimitiveInstr
        instr.id = Prims.Comp.USE
    }
).parallelWithoutDeepCopy()