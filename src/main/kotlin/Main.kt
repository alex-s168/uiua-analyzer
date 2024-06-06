package me.alex_s168.uiua

import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.opt.optInlineCUse
import me.alex_s168.uiua.ir.opt.optRemUnused
import me.alex_s168.uiua.ir.putBlock
import me.alex_s168.uiua.ir.toIr
import me.alex_s168.uiua.mlir.emitMLIR

fun loadRes(file: String): String? =
    object {}.javaClass.classLoader.getResourceAsStream(file)?.reader()?.readText()

private fun IrBlock.findAllRequiredCompile(dosth: (IrBlock) -> Unit): Set<IrBlock> {
    val list = mutableSetOf<IrBlock>()

    fun rec(block: IrBlock) {
        dosth(block)
        if (block !in list) {
            list.add(block)
            block.instrs.forEach {
                if (it.instr is PushFnRefInstr) {
                    val fn = block.ref(it.instr.fn)!!
                    rec(fn)
                }
            }
        }
    }
    rec(this)

    return list
}

fun main() {
    val test = loadRes("test.uasm")!!
    val assembly = Assembly.parse(test)

    val blocks = assembly.functions.toIr()

    val expanded = blocks["fn"]!!.expandFor(listOf(Types.array(Types.int)), blocks::putBlock)

    val compile = blocks[expanded]!!.findAllRequiredCompile() {
        it.optInlineCUse()
        it.optRemUnused()
    }

    val out = StringBuilder()
    out.append(loadRes("runtime.mlir")!!)
    out.append("\n\n")

    compile.forEach {
        out.append(it.emitMLIR())
        out.append("\n\n")
    }

    println(out)
}