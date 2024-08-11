package me.alex_s168.uiua

import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.analysis.lifetimes
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

object Inline {
    val all = { block: IrBlock -> true }
    val none = { block: IrBlock -> false }
    fun below(max: Int) =
        { block: IrBlock -> block.instrs.size < max }
}

val inlineConfig = Inline.all
val unfilledLoadBoundsCheck = false

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
        remUnused.generic(),
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

        fixArgArrays.generic(),
        inlineCUse.generic(),
        lifetimes.generic(),
        remClone.generic(),

        lowerClone.generic(),
        lowerShape.generic(),
        lowerLen.generic(),
        // boundsChecking.generic(),  // TODO
        evalDim.generic(),
        remUnused.generic(), // before materialize!
        remArrMat.generic(),
        lowerMaterialize.generic(),

        fixArgArrays.generic(),
        inlineCUse.generic(),

        constantTrace.generic(),
        funcInline.generic(),
        switchDependentCodeMovement.generic(),
        remUnused.generic(),
        dce.generic(),
    ))

    val passes2 = passPipeline(listOf(
        remUnused.generic(),
        switchDependentCodeMovement.generic(),
        remUnused.generic(),
        remComments.generic(),
        argRem.generic(),
        switchDependentCodeMovement.generic(),
        switchIndependentTrailingCodeMovement.generic(),
        remUnused.generic(),
        emptyArrayOpsRemove.generic(),
        dce.generic(),
        remUnused.generic(),
    ))

    val compile = File(".out.uac").printWriter().use { file ->
        blocks[expanded]!!.findAllRequiredCompile {
            // only reorder if you know what you are doing!

            val res = runCatching {
                passes(it, blocks::putBlock)
                dse(expanded, blocks)
                passes2(it, blocks::putBlock)
                dse(expanded, blocks)
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
    val llvmLower = true
    val enableBufferDealloc = false

    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, control-flow-sink, inline, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, ownership-based-buffer-deallocation, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, canonicalize, loop-invariant-code-motion, control-flow-sink, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)"
    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, control-flow-sink, inline, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-parallel, canonicalize, one-shot-bufferize, ownership-based-buffer-deallocation, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, canonicalize, convert-scf-to-cf, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, convert-to-llvm, canonicalize, reconcile-unrealized-casts)"
    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, control-flow-sink, inline, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, scf-forall-to-for, canonicalize, control-flow-sink)"
    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, control-flow-sink, inline, canonicalize, loop-invariant-code-motion, scf-for-loop-specialization, scf-forall-to-parallel, canonicalize, control-flow-sink, one-shot-bufferize, ownership-based-buffer-deallocation, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, convert-scf-to-cf, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, convert-to-llvm, reconcile-unrealized-casts)"
    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, control-flow-sink, inline, canonicalize, loop-invariant-code-motion, scf-for-loop-specialization, scf-forall-to-parallel, canonicalize, control-flow-sink)"
    // --pass-pipeline="builtin.module(func(cse, canonicalize), inline, sccp, sroa, inline, sccp, control-flow-sink, one-shot-bufferize, ownership-based-buffer-deallocation, inline, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, sccp, sroa, inline, convert-tensor-to-linalg, convert-linalg-to-affine-loops, func.func(affine-parallelize, affine-loop-fusion, affine-loop-invariant-code-motion, affine-loop-tile, affine-super-vectorize, affine-loop-unroll, affine-scalrep), lower-affine, convert-scf-to-cf, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, convert-to-llvm, reconcile-unrealized-casts)"
    // --pass-pipeline=builtin.module(func(cse, canonicalize), inline, sccp, sroa, one-shot-bufferize, convert-bufferization-to-memref, convert-tensor-to-linalg, convert-linalg-to-affine-loops, func.func(affine-parallelize), affine-loop-fusion, func.func(affine-loop-invariant-code-motion, affine-loop-tile, affine-super-vectorize, affine-loop-unroll, affine-scalrep), lower-affine, async-parallel-for, convert-scf-to-cf, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma,$llvmLowerStr reconcile-unrealized-casts)

    // -pass-pipeline=builtin.module(func(cse, canonicalize), sccp, sroa, inline, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, ownership-based-buffer-deallocation, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), buffer-deallocation-simplification, func.func(buffer-loop-hoisting), convert-bufferization-to-memref, control-flow-sink, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, canonicalize, loop-invariant-code-motion, control-flow-sink, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)
    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, inline, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, ownership-based-buffer-deallocation, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), buffer-deallocation-simplification, func.func(buffer-loop-hoisting), convert-bufferization-to-memref, control-flow-sink, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)"
    // --pass-pipeline="builtin.module(func(cse, canonicalize), sccp, sroa, inline, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), ownership-based-buffer-deallocation, buffer-deallocation-simplification, func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, control-flow-sink, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)"
    val bufferDealloc = if (enableBufferDealloc) "ownership-based-buffer-deallocation, buffer-deallocation-simplification, " else ""
    val mlirPipeline = if (llvmLower) "-pass-pipeline=builtin.module(func(cse, canonicalize), sccp, sroa, inline, symbol-dce, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), ${bufferDealloc}func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, control-flow-sink, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)"
    else "-pass-pipeline=builtin.module(func(cse, canonicalize), sccp, sroa, inline, symbol-dce, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), ${bufferDealloc}func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, control-flow-sink, canonicalize)"
    val mlirOptFlags = listOf(mlirPipeline)
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
        ?.run(listOf(clang, "-c", "-O3", "-march=native", "-o", outObj) + clangFlags + outLlc)
        ?.let { println("Generated .out.o") }
        ?: run {
            println("Could not compile to object file!")
            exitProcess(1)
        }
}