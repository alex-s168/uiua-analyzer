package me.alex_s168.uiua

import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.lower.*
import me.alex_s168.uiua.ir.opt.*
import me.alex_s168.uiua.ir.transform.*
import me.alex_s168.uiua.mlir.emitMLIR
import me.alex_s168.uiua.mlir.emitMLIRFinalize
import java.io.File
import kotlin.random.Random
import kotlin.random.nextULong
import kotlin.system.exitProcess

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
                    val fn = block.ref[it.instr.fn]!!
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

    val expanded = blocks["fn"]!!.expandFor(listOf(
        Types.array(Types.array(Types.int)),
    ), blocks::putBlock)
    blocks[expanded]!!.private = false

    fun Pass<Unit>.generic(): Pass<(IrBlock) -> Unit> =
        Pass(name) { b, _ -> internalRun(b, Unit) }

    fun Pass<(IrBlock) -> Unit>.generic() =
        this

    val passes = passPipeline(listOf(
        lowerDup.generic(),
        lowerOver.generic(),
        lowerFlip.generic(),
        inlineCUse.generic(),
        // remUnused.generic(),
        lowerPervasive.generic(),
        lowerUnShape.generic(),
        lowerEach.generic(),
        lowerRows.generic(),
        comptimeReduceEval.generic(),
        lowerReduce.generic(),
        lowerRange.generic(),
        lowerReverse.generic(),
        lowerFix.generic(),
        lowerPick.generic(),
        lowerUndoPick.generic(),
        lowerBox.generic(),
        lowerUnBox.generic(),
        lowerArrImm.generic(),
        boxConv.generic(),
        lowerBoxLoad.generic(),
        lowerBoxStore.generic(),
        lowerBoxCreate.generic(),
        lowerBoxDestroy.generic(),
        lowerClone.generic(),
        lowerShape.generic(),
        lowerLen.generic(),
        evalDim.generic(),
        // remUnused.generic(), // before materialize!
        // remArrMat.generic(),
        lowerMaterialize.generic(),
        fixArgArrays.generic(),
        inlineCUse.generic(),
        // remUnused.generic(),
    ))

    val compile = File(".out.uac").printWriter().use { file ->
        blocks[expanded]!!.findAllRequiredCompile {
            // only reorder if you know what you are doing!

            val res = runCatching {
                passes(it, blocks::putBlock)
            }

            file.println(it)
            file.println()

            res.onFailure {
                println("in apply pass pipeline")
                throw it
            }
        }
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

    val mlirOpt = "/home/alex/llvm-project/build/bin/mlir-opt"
    val mlirTranslate = "/home/alex/llvm-project/build/bin/mlir-translate"
    val clang = "clang"
    val llvmLower = false

    // TODO: add --ownership-based-buffer-deallocation back (after --one-shot-bufferize)
    val llvmLowerStr = if (llvmLower) " convert-to-llvm," else ""
    val mlirOptFlags = listOf("--pass-pipeline=builtin.module(func(cse, canonicalize), inline, sccp, sroa, one-shot-bufferize, convert-bufferization-to-memref, convert-tensor-to-linalg, convert-linalg-to-affine-loops, func.func(affine-parallelize), affine-loop-fusion, func.func(affine-loop-invariant-code-motion, affine-loop-tile, affine-super-vectorize, affine-loop-unroll, affine-scalrep), lower-affine, async-parallel-for, convert-scf-to-cf, mem2reg, math-uplift-to-fma,$llvmLowerStr reconcile-unrealized-casts)")
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
            exitProcess(1)
        }
}