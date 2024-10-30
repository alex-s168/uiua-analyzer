package me.alex_s168.uiua

import blitz.collections.LightCache
import blitz.collections.RefVec
import blitz.collections.caching
import blitz.collections.gather
import blitz.parse.comb2.ParseResult
import me.alex_s168.uiua.ast.ASTRoot
import me.alex_s168.uiua.ast.genGraph
import me.alex_s168.uiua.ir.*
import me.alex_s168.uiua.ir.analysis.lifetimes
import me.alex_s168.uiua.ir.lower.*
import me.alex_s168.uiua.ir.opt.*
import me.alex_s168.uiua.ir.transform.*
import me.alex_s168.uiua.mlir.emitMLIR
import me.alex_s168.uiua.mlir.emitMLIRFinalize
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import kotlin.random.nextULong
import kotlin.system.exitProcess
import kotlin.system.measureTimeMillis

fun anonFnName(): String =
    "_\$anon_${Random.nextULong()}"

fun loadRes(file: String): String? =
    object {}.javaClass.classLoader.getResourceAsStream(file)?.reader()?.readText()

class CallerInstrsCache(initCap: Int = 0) {
    val cache = LightCache.new<Int, Sequence<Pair<IrBlock, IrInstr>>>(initCap)

    fun get(blk: IrBlock): Sequence<Pair<IrBlock, IrInstr>> =
        cache.getOrPut(blk.uid) {
            Analysis(blk).callerInstrs(::get).caching()
        }
}

object Inline {
    val all = { block: IrBlock -> true }
    val none = { block: IrBlock -> false }
    fun lte(max: Int) =
        { block: IrBlock -> block.instrs.size <= max }
}

object UnrollLoop {
    val all = { _: IrBlock, _: Int -> true }
    fun inlineCfg(i: (IrBlock) -> Boolean) =
        { block: IrBlock, _: Int -> i(block) }
    fun lteInlineCfg(maxLoopIters: Int, i: (IrBlock) -> Boolean) =
        { block: IrBlock, c: Int -> c <= maxLoopIters && i(block) }
    fun sumLte(max: Int) =
        { block: IrBlock, c: Int -> block.instrs.size * c <= max }
}

val inlineConfig = Inline.none // TODO: change back
val unfilledLoadBoundsCheck = false
val fullUnrollLoop = UnrollLoop.sumLte(64)
val boundsChecking = true // pick and unpick bounds checking
val mlirComments = true
val dontCareOpsBeforePanic = true
val additionalDebugInstrs = false // don't really work
val uacPrintSpans = true
val debugVerify = false

// TODO: don't use scf.parallel if it can't safely be executed in parallel

fun main() {
    val test = loadRes("test.uasm")!!
    val assembly = Assembly.parse(test)

    val astNodes = mutableListOf<ASTRoot>()
    val blocks = assembly.functions.toIr(astNodes)

    File(".ast.dot").writer().use { w ->
        w.append(astNodes.genGraph())
    }

    val old = blocks.keys.toList()
    val exported = listOf(blocks["fn"]!!.expandFor(listOf(
        //Types.array(Types.array(Types.int)),
    ), blocks::putBlock))
    old.forEach(blocks::remove)

    exported.forEach { blocks[it]!!.private = false }

    File(".in.uac").writer().use { w ->
        blocks.values.forEach {
            w.append(it.toString())
            w.append("\n\n")
        }
    }

    fun Pass<Unit>.generic(): Pass<(IrBlock) -> Unit> =
        Pass(name) { b, _ -> internalRun(b, Unit) }

    fun GlobalPass<Unit>.generic(): GlobalPass<(IrBlock) -> Unit> =
        GlobalPass(name) { b, _ -> internalRun(b, Unit) }

    fun Pass<(IrBlock) -> Unit>.generic() =
        this

    fun GlobalPass<(IrBlock) -> Unit>.generic() =
        this

    fun <A> GlobalPass<A>.setArg(v: A): GlobalPass<Unit> =
        GlobalPass(name) { b, _ -> this@setArg.internalRun(b, v) }

    // TODO: fix argrem

    val passes = listOf(
        lowerUnTranspose.generic(),
        lowerTranspose.generic(),
        lowerJoin.generic(),
        lowerUnCouple.generic(),
        lowerReduceDepth.generic(),
        lowerDup.generic(),
        lowerOver.generic(),
        lowerFlip.generic(),
        inlineCUse.generic(),
        remUnused.generic(),
        lowerPervasive.generic(),
        lowerUnShape.generic(),
        lowerReshape.generic(),
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
        lowerDeshape.generic(),

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
        unrollLoop.generic(),

        lowerFill.generic(),
        fixFnTypes.generic(),

        //oneBlockOneCaller.generic(),
        //constantTrace.generic(),
        //funcInline.generic(),
        switchDependentCodeMovement.generic(),
        remUnused.generic(),
        dce.generic(),
        dse.setArg(exported).generic(),
        remUnused.generic(),
        switchDependentCodeMovement.generic(),
        remUnused.generic(),
        remComments.generic(),
        //oneBlockOneCaller.generic(),
        //argRem.generic(),
        switchDependentCodeMovement.generic(),
        //oneBlockOneCaller.generic(),
        //constantTrace.generic(),
        funcInline.generic(),
        funcInline.generic(),
        funcInline.generic(),
        funcInline.generic(),
        switchIndependentTrailingCodeMovement.generic(),
        dse.setArg(exported).generic(),
        licm.generic(),
        loopUnswitch.generic(),
        remUnused.generic(),
        emptyArrayOpsRemove.generic(),
        dce.generic(),
        remUnused.generic(),
        funcInline.generic(),
        dce.generic(),
        remUnused.generic(),
        deadRetsRem.generic(),
        deadRetsRem.generic(),
        deadRetsRem.generic(),
        remUnused.generic(),
        dce.generic(),
        //identicalSwitchRem.generic(), // TODO: fix
        //argRem.generic(),
        remUnused.generic(),
        dce.generic(),
        //argRem.generic(),
        remUnused.generic(),
        dce.generic(),
        dse.setArg(exported).generic(),
    )

    val compile = File(".out.uac").printWriter().use { file ->
        val res = runCatching {
            passes.apply(blocks)
        }

        blocks.values.forEach {
            file.println(it)
            file.println()
        }

        res.onFailure {
            println("================================================================================")
            println("in apply pass pipeline")
            throw it
        }

        blocks.values.toSet()
    }

    println("emitting MLIR...")

    val out = StringBuilder()
    out.append(loadRes("runtime.mlir")!!)
    out.append("\n\n")

    compile.forEach { block ->
        out.append(block.emitMLIRFinalize(block.emitMLIR { loc ->
            listOf(assembly.spans
                .gather(loc.uasmSpanIdc)
                .joinToString()
                .let { "// source: $it" })
        }))
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