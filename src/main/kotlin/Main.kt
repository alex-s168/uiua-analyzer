package me.alex_s168.uiua

import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.opt.optInlineCUse
import me.alex_s168.uiua.ir.opt.optRemUnused
import me.alex_s168.uiua.ir.putBlock
import me.alex_s168.uiua.ir.toIr
import me.alex_s168.uiua.ir.transform.*
import me.alex_s168.uiua.mlir.emitMLIR
import java.io.File
import kotlin.math.exp
import kotlin.random.Random
import kotlin.random.nextULong

fun anonFnName(): String =
    "_\$anon_${Random.nextULong()}"

fun loadRes(file: String): String? =
    object {}.javaClass.classLoader.getResourceAsStream(file)?.reader()?.readText()

private fun IrBlock.findAllRequiredCompile(dosth: (IrBlock) -> Unit): Set<IrBlock> {
    val list = mutableSetOf<IrBlock>()

    fun rec(block: IrBlock) {
        if (list.none { it.name == block.name }) {
            dosth(block)
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

    val expanded = blocks["fn"]!!.expandFor(listOf(Types.array(Types.int)/*, Types.array(Types.int)*/), blocks::putBlock)
    blocks[expanded]!!.private = false

    val compile = blocks[expanded]!!.findAllRequiredCompile {
        // only reorder if you know what you are doing!
        it.expandStackOps()
        it.optInlineCUse()
        it.optRemUnused()
        it.lowerReduce(blocks::putBlock)
        it.expandBoxes()
        it.expandArrays()
        it.lowerBoxesToArrays()
        it.lowerSimple()

        println(it)
        println()
    }

    val out = StringBuilder()
    out.append(loadRes("runtime.mlir")!!)
    out.append("\n\n")

    compile.forEach {
        out.append(it.emitMLIR())
        out.append("\n\n")
    }

    val inMlir = ".in.mlir"
    val optMlir = ".opt.mlir"
    val outLlc = ".out.llc"
    val outObj = ".out.o"

    File(inMlir).writeText(out.toString())

    val mlirOpt = "mlir-opt"
    val mlirTranslate = "mlir-translate"
    val clang = "clang"

    // TODO: add --ownership-based-buffer-deallocation back (after --one-shot-bufferize)
    val mlirOptFlags = "--one-shot-bufferize --convert-bufferization-to-memref --convert-tensor-to-linalg --convert-linalg-to-affine-loops --lower-affine -convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts "
    val mlirTranslateFlags = "--mlir-to-llvmir"
    val clangFlags = "-x ir"

    try {
        require(Runtime.getRuntime().exec("$mlirOpt -o $optMlir $inMlir $mlirOptFlags").waitFor() == 0)
        require(Runtime.getRuntime().exec("$mlirTranslate -o $outLlc $optMlir $mlirTranslateFlags").waitFor() == 0)
        require(Runtime.getRuntime().exec("$clang -c -O3 -o $outObj $clangFlags $outLlc").waitFor() == 0)

        println("Generated .out.o")
    } catch (e: Exception) {
        println("Could not compile to object file!")
        println()
        println("Generated MLIR:")
        println("=================")
        println()
        println(out.toString())
    }
}