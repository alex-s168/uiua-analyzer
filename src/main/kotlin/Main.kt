package me.alex_s168.uiua

import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.opt.optInlineCUse
import me.alex_s168.uiua.ir.opt.optRemUnused
import me.alex_s168.uiua.ir.putBlock
import me.alex_s168.uiua.ir.toIr
import me.alex_s168.uiua.mlir.emitMLIR
import kotlin.math.exp

fun loadRes(file: String): String? =
    object {}.javaClass.classLoader.getResource(file)?.readText()

private fun emitMlirRec(dest: StringBuilder, done: MutableSet<IrBlock>, block: IrBlock) {
    block.instrs.forEach {
        if (it.instr is PushFnRefInstr) {
            val fn = block.ref(it.instr.fn)!!
            emitMlirRec(dest, done, fn)
        }
    }

    if (block !in done) {
        dest.append(block.emitMLIR())
        dest.append("\n\n")
        done.add(block)
    }
}

fun main() {
    val test = loadRes("test.uasm")!!
    val assembly = Assembly.parse(test)

    val blocks = assembly.functions.toIr()

    val expanded = blocks["fn"]!!.expandFor(listOf(Types.array(Types.int)), blocks::putBlock)

    blocks.forEach { (_, v) ->
        v.optInlineCUse()
        v.optRemUnused()
    }

    val out = StringBuilder()
    out.append(loadRes("runtime.mlir")!!)
    out.append("\n\n")

    val done = mutableSetOf<IrBlock>()
    emitMlirRec(out, done, blocks[expanded]!!)

    println(out)
}