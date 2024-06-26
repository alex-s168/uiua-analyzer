package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.ir.optAwayPass

val remUnused = optAwayPass(
    "rem unused",
    { true },
    { a ->
        outs.size > 0 && outs.all { a.unused(it) }
    }
)