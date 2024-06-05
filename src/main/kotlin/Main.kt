package me.alex_s168.uiua

import blitz.collections.stringify
import blitz.io.Path
import blitz.io.read
import blitz.str.flattenToString
import me.alex_s168.uiua.ir.opt.optInlineCUse
import me.alex_s168.uiua.ir.opt.optRemUnused
import me.alex_s168.uiua.ir.putBlock
import me.alex_s168.uiua.ir.toIr

fun main() {
    val test = Path.of("C:\\Users\\Alexander.Nutz\\uiua-analyzer\\test.uasm")
        .getFile()
        .read()
        .stringify()
        .flattenToString()
    val assembly = Assembly.parse(test)

    val blocks = assembly.functions.toIr()

    blocks["fn"]!!.expandFor(listOf(Types.array(Types.int)), blocks::putBlock)

    blocks.forEach { (_, v) ->
        v.optInlineCUse()
        v.optRemUnused()

        println(v)
        println()
    }
}