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
    val old = blocks.keys.toList()
    val expanded = blocks["fn"]!!.expandFor(listOf(
        // Types.array(Types.array(Types.int)),
    ), blocks::putBlock)
    old.forEach(blocks::remove)
    blocks[expanded]!!.private = false

    fun Pass<Unit>.generic(): Pass<(IrBlock) -> Unit> =
        Pass(name) { b, _ -> internalRun(b, Unit) }

    fun Pass<(IrBlock) -> Unit>.generic() =
        this

    val passes = listOf(
        lowerDup.generic(),
        lowerOver.generic(),
        lowerFlip.generic(),
        inlineCUse.generic(),
        remUnused.generic(),
        lowerPervasive.generic(),
        lowerUnShape.generic(),
        lowerEach.generic(),
        lowerTable.generic(),
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
        lowerReshape.generic(),

        fixArgArrays.generic(),
        inlineCUse.generic(),
        lifetimes.generic(),
        remClone.generic(),

        lowerClone.generic(),
        lowerShape.generic(),
        lowerLen.generic(),
        // boundsChecking.generic(),  // TODO only for pick
        evalDim.generic(),
        remUnused.generic(), // before materialize!
        remArrMat.generic(),
        lowerMaterialize.generic(),

        fixArgArrays.generic(),
        inlineCUse.generic(),
    )
    // lower fill happens here
    val passes2 = listOf(
        oneBlockOneCaller.generic(),
        constantTrace.generic(),
        funcInline.generic(),
        switchDependentCodeMovement.generic(),
        remUnused.generic(),
        dce.generic(),
    )
    // dse happens here
    val passes3 = listOf(
        remUnused.generic(),
        switchDependentCodeMovement.generic(),
        remUnused.generic(),
        remComments.generic(),
        argRem.generic(),
        switchDependentCodeMovement.generic(),
        oneBlockOneCaller.generic(),
        constantTrace.generic(),
        funcInline.generic(),
        switchIndependentTrailingCodeMovement.generic(),
        remUnused.generic(),
        emptyArrayOpsRemove.generic(),
        dce.generic(),
        remUnused.generic(),
    )

    val compile = File(".out.uac").printWriter().use { file ->
        val res = runCatching {
            fun apply(pipe: List<Pass<(IrBlock) -> Unit>>) {
                val alreadyDone = mutableSetOf<IrBlock>()

                while (true) {
                    val old = blocks.values.toList()
                    val todo = (old - alreadyDone)
                    if (todo.isEmpty()) break
                    todo.forEach { b ->
                        pipe.forEach {
                            it.run(b, blocks::putBlock)
                        }
                    }
                    alreadyDone.addAll(old)
                }
            }

            apply(passes)
            blocks.values.toList().forEach { lowerFill.run(it) }
            blocks.values.toList().forEach { Analysis(it).updateFnType() }
            apply(passes2)
            dse(expanded, blocks)
            apply(passes3)
            dse(expanded, blocks)

            blocks.values.forEach {
                file.println(it)
                file.println()
            }
        }

        res.onFailure {
            println("in apply pass pipeline")
            throw it
        }

        blocks.values.toSet()
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

    // TODO: remove fill emit logic

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