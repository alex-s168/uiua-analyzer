package me.alex_s168.uiua.ir.opt

import me.alex_s168.uiua.ir.IrBlock

fun IrBlock.optRemUnused() {
    instrs.removeIf {
        it.outs.all { !varUsed(it) }
    }
}