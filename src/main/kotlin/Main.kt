package me.alex_s168.uiua

import me.alex_s168.uiua.ast.ASTRoot
import me.alex_s168.uiua.ast.genGraph
import me.alex_s168.uiua.ast.printAst
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

inline fun <reified R> Any?.cast(): R? =
    this?.let { if (it is R) it else null }

fun anonFnName(): String =
    "_\$anon_${Random.nextULong()}"

fun loadRes(file: String): String? =
    object {}.javaClass.classLoader.getResourceAsStream(file)?.reader()?.readText()

fun <T> Iterable<Iterable<T>>.intersections(dest: MutableList<T> = mutableListOf()): MutableList<T> =
    reduce { acc, li -> acc.intersect(li) }
        .forEach { dest += it }
        .let { dest }

fun <T> Iterable<T>.removeAtIndexes(idc: Iterable<Int>, dest: MutableList<T> = mutableListOf()): MutableList<T> =
    filterIndexedTo(dest) { index, _ -> index !in idc }

fun <T> List<T>.gather(idc: Iterable<Int>): MutableList<T> {
    val dest = mutableListOf<T>()
    idc.forEach {
        dest += get(it)
    }
    return dest
}

fun <T> List<T>.before(idx: Int): List<T> =
    take(idx)

fun <T> List<T>.after(idx: Int): List<T> =
    drop(idx + 1)

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

val inlineConfig = Inline.all
val unfilledLoadBoundsCheck = false
val fullUnrollLoop = UnrollLoop.sumLte(64)
val boundsChecking = true // pick and unpick bounds checking
val mlirComments = true
val dontCareOpsBeforePanic = true
val additionalDebugInstrs = false // don't really work
val uacPrintSpans = true

// TODO: don't use scf.parallel if it can't safely be executed in parallel

fun main() {
    val test = loadRes("test.uasm")!!
    val assembly = Assembly.parse(test)

    val astNodes = mutableListOf<ASTRoot>()
    val blocks = assembly.functions.toIr(astNodes)

    /*astNodes.forEach {
        println(it.functionName)
        it.children.printAst(indent = 2)
    }*/

    File(".ast.dot").writer().use { w ->
        w.append(astNodes.genGraph())
    }

    val old = blocks.keys.toList()
    val expanded = blocks["fn"]!!.expandFor(listOf(
        Types.array(Types.array(Types.int)),
    ), blocks::putBlock)
    old.forEach(blocks::remove)
    blocks[expanded]!!.private = false

    File(".in.uac").writer().use { w ->
        blocks.values.forEach {
            w.append(it.toString())
            w.append("\n\n")
        }
    }

    fun Pass<Unit>.generic(): Pass<(IrBlock) -> Unit> =
        Pass(name) { b, _ -> internalRun(b, Unit) }

    fun Pass<(IrBlock) -> Unit>.generic() =
        this

    val passes = listOf(
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
    )

    // lower fill happens here
    val passes2 = listOf(
        oneBlockOneCaller.generic(),
        constantTrace.generic(),
        //funcInline.generic(),
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
        funcInline.generic(),
        funcInline.generic(),
        funcInline.generic(),
        switchIndependentTrailingCodeMovement.generic(),
    )
    val passes4 = listOf(
        licm.generic(),
        loopUnswitch.generic(),
        remUnused.generic(),
        oneBlockOneCaller.generic(),
        emptyArrayOpsRemove.generic(),
        dce.generic(),
        remUnused.generic(),
    )
    val passes5 = listOf(
        funcInline.generic(),
        dce.generic(),
        remUnused.generic(),
        deadRetsRem.generic(),
        deadRetsRem.generic(),
        deadRetsRem.generic(),
        remUnused.generic(),
        dce.generic(),
        //identicalSwitchRem.generic(), // TODO: fix
    )

    val passes6 = listOf(
        argRem.generic(),
        remUnused.generic(),
        dce.generic(),
        argRem.generic(),
        remUnused.generic(),
        dce.generic(),
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
            apply(passes4)
            apply(passes5)
            apply(passes6)
            dse(expanded, blocks)
        }

        blocks.values.forEach {
            file.println(it)
            file.println()
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

    // TODO: remove fill emit logic

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