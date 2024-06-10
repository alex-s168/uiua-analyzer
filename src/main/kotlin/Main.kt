package me.alex_s168.uiua

import me.alex_s168.uiua.ir.IrBlock
import me.alex_s168.uiua.ir.opt.optInlineCUse
import me.alex_s168.uiua.ir.opt.optRemUnused
import me.alex_s168.uiua.ir.putBlock
import me.alex_s168.uiua.ir.toIr
import me.alex_s168.uiua.ir.transform.*
import me.alex_s168.uiua.mlir.emitMLIR
import me.alex_s168.uiua.mlir.emitMLIRFinalize
import java.io.File
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

            var idx = 0
            while (idx < block.instrs.size) {
                val it = block.instrs[idx]
                if (it.instr is PushFnRefInstr) {
                    val fn = block.ref(it.instr.fn)!!
                    val oldSize = block.instrs.size
                    rec(fn)
                    idx += block.instrs.size - oldSize
                }
                idx ++
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
        fun IrBlock.basicOpt() {
            optInlineCUse()
            optRemUnused()
        }

        it.expandStackOps()
        it.basicOpt()
        it.lowerPervasive(blocks::putBlock)
        it.lowerRows(blocks::putBlock)
        it.lowerReduce(blocks::putBlock)
        it.lowerRange(blocks::putBlock)
        it.lowerReverse(blocks::putBlock)
        it.expandBoxes()
        it.expandArrays()
        it.lowerBoxesToArrays()
        it.lowerSimple()
        it.basicOpt()

        println(it)
        println()
    }

    val out = StringBuilder()
    out.append(loadRes("runtime.mlir")!!)
    out.append("\n\n")

    compile.forEach {
        out.append(it.emitMLIRFinalize(it.emitMLIR()))
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
    val llvmLower = true

    // TODO: add --ownership-based-buffer-deallocation back (after --one-shot-bufferize)
    val llvmLowerStr = if (llvmLower) " convert-to-llvm," else ""
    val mlirOptFlags = listOf("--pass-pipeline=builtin.module(func(cse, canonicalize), inline, sccp, sroa, one-shot-bufferize, convert-bufferization-to-memref, convert-tensor-to-linalg, convert-linalg-to-affine-loops, lower-affine, convert-scf-to-cf, mem2reg,$llvmLowerStr reconcile-unrealized-casts)")
    val mlirTranslateFlags = listOf("--mlir-to-llvmir")
    val clangFlags = listOf("-x", "ir")

    fun Unit.run(cmd: List<String>): Unit? {
        println("\$ ${cmd.joinToString(" ")}")
        if (Runtime.getRuntime().exec(cmd.toTypedArray()).waitFor() == 0)
            return Unit
        return null
    }

    Unit.run(listOf(mlirOpt, "-o", optMlir, inMlir) + mlirOptFlags)
        ?.run(listOf(mlirTranslate, "-o", outLlc, optMlir) + mlirTranslateFlags)
        ?.run(listOf(clang, "-c", "-O3", "-o", outObj) + clangFlags + outLlc)
        ?.let { println("Generated .out.o") }
        ?: run {
            println("Could not compile to object file!")
            println()
            println("Generated MLIR:")
            println("=================")
            println()
            println(out.toString())
        }
}