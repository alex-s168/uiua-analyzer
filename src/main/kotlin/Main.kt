package me.alex_s168.uiua

import blitz.Either
import blitz.collections.*
import blitz.flatten
import blitz.parse.comb2.ParseResult
import blitz.then
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
import java.io.Writer
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.random.Random
import kotlin.random.nextULong
import kotlin.system.exitProcess
import kotlin.system.measureTimeMillis

// TODO: use block uid for call instrs
// TODO: fast contains on RefVec
// TODO: use ids for primitives
// TODO: RefVec remove methods (remove range, remove if, ...)

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

// TODO: move to blitz

fun <T, C: Vec<T>> Sequence<T>.toVec(dest: C): C {
    forEach { dest.pushBack(it) }
    return dest
}

fun <T> Sequence<T>.toVec() = toVec(RefVec())

fun <T> Vec<T>.fastToMutableList() =
    MutableList(this.size) { this[it] }

fun <T, R: Any> Iterable<T>.mapOrNull(fn: (T) -> R?): RefVec<R>? {
    val out = RefVec<R>()
    forEach {
        val v = fn(it)
        if (v == null)
            return null
        out.pushBack(v)
    }
    return out
}

fun <A: Any, B: Any, RA: Any> Either<A, B>.mapAOrNull(fn: (A) -> RA?): Either<RA, B>? {
    if (a != null) {
        val v = fn(a!!) ?: return null
        return Either.ofA(v)
    }
    return Either.ofB(b!!)
}

fun <A: Any, B: Any, RB: Any> Either<A, B>.mapBOrNull(fn: (B) -> RB?): Either<A, RB>? {
    if (b != null) {
        val v = fn(b!!) ?: return null
        return Either.ofB(v)
    }
    return Either.ofA(a!!)
}

fun <A: Any, B: Any, CA: Collection<A>, CB: Collection<B>> Either<CA, CB>.inside(): RefVec<Either<A, B>> =
    RefVec<Either<A, B>>(flatten().size).also { r ->
        this.then(
            { it.forEach { r.pushBack(Either.ofA(it)) } },
            { it.forEach { r.pushBack(Either.ofB(it)) } },
        )
    }

fun <A: Any, B: Any, CA: Collection<A>, CB: Collection<B>, R> Either<CA, CB>.insideFlatMap(fnA: (A) -> R, fnB: (B) -> R): RefVec<R> =
    RefVec<R>(flatten().size).also { r ->
        this.then(
            { it.forEach { r.pushBack(fnA(it)) } },
            { it.forEach { r.pushBack(fnB(it)) } },
        )
    }

fun <A: C, B: C, CA: Collection<A>, CB: Collection<B>, C: Any> Either<CA, CB>.insideFlatten(): RefVec<C> =
    RefVec<C>(flatten().size).also { r ->
        this.then(
            { it.forEach { r.pushBack(it) } },
            { it.forEach { r.pushBack(it) } },
        )
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

var log = { str: String ->
    println(str)
}

val inlineConfig = Inline.lte(64)
val unfilledLoadBoundsCheck = false
val fullUnrollLoop = UnrollLoop.sumLte(64)
val boundsChecking = true // pick and unpick bounds checking
val mlirComments = true
val dontCareOpsBeforePanic = true
val additionalDebugInstrs = false // don't really work
val uacPrintSpans = true
val debugVerify = false

// TODO: don't use scf.parallel if it can't safely be executed in parallel

class ConcurrentLogger(private val dest: Writer) {
    private val logwrcache = ThreadLocal.withInitial<String?> { null }
    private val logwrlock = ReentrantLock()
    fun log(txt: String) {
        if (logwrlock.isLocked && (logwrcache.get()?.length ?: 0) < 1000) {
            logwrcache.set("${logwrcache.get()}$txt\n")
        } else {
            logwrlock.withLock {
                logwrcache.get()?.let {
                    if (it != "null")
                        dest.appendLine(it)
                    logwrcache.set(null)
                }
                if (txt != "null")
                    dest.appendLine(txt)
            }
        }
    }
}

fun main() {
    File(".log.txt").bufferedWriter().use { logwriter ->
        log = ConcurrentLogger(logwriter)::log

        val test = loadRes("test.uasm")!!
        val assembly = Assembly.parse(test)

        val astNodes = mutableListOf<ASTRoot>()
        val blocks = assembly.functions.toIr(astNodes)

        File(".ast.dot").writer().use { w ->
            w.append(astNodes.genGraph())
        }

        val old = blocks.keys.toList()
        val exported = listOf(
            blocks.find("fn")!!.expandFor(
                listOf(
                    //Types.array(Types.array(Types.int)),
                ), blocks::putBlock
            )
        )
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
            lowerDeshape.generic(),
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

            lowerFill.generic(),
            fixFnTypes.generic(),

            globalPrint.setArg { File(".preOpt.uac").printWriter().use(it) }.generic(),

            //verifyBlock.generic(),

            fixArgArrays.generic(),
            inlineCUse.generic(),
            unrollLoop.generic(), // TODO: fix

            //oneBlockOneCaller.generic(),
            //constantTrace.generic(),
            //funcInline.generic(),

            // TODO: !!!! fix switch move

            //switchDependentCodeMovement.generic(),
            fixFnTypes.generic(),

            remUnused.generic(),
            dce.generic(),
            dse.setArg(exported).generic(),
            remUnused.generic(),
            //switchDependentCodeMovement.generic(),
            fixFnTypes.generic(),
            remUnused.generic(),
            //remComments.generic(),
            //oneBlockOneCaller.generic(),
            //argRem.generic(),
            //switchDependentCodeMovement.generic(),
            fixFnTypes.generic(),
            //oneBlockOneCaller.generic(),
            //constantTrace.generic(),
            funcInline.generic(),
            funcInline.generic(),
            funcInline.generic(),
            funcInline.generic(),
            //switchIndependentTrailingCodeMovement.generic(),
            fixFnTypes.generic(),
            dse.setArg(exported).generic(),
            //licm.generic(),
            //loopUnswitch.generic(),
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
            //fixFnTypes.generic(),
            //argRem.generic(),
            remUnused.generic(),
            dce.generic(),
            //argRem.generic(),
            remUnused.generic(),
            dce.generic(),
            dse.setArg(exported).generic(),
            fixFnTypes.generic(),
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
                log("================================================================================")
                log("in apply pass pipeline")
                throw it
            }

            blocks.values.toSet()
        }

        log("emitting MLIR...")

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

        val bufferDealloc =
            if (enableBufferDealloc) "ownership-based-buffer-deallocation, buffer-deallocation-simplification, " else ""
        val mlirPipeline =
            if (llvmLower) "-pass-pipeline=builtin.module(func(cse, canonicalize), sccp, sroa, inline, symbol-dce, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), ${bufferDealloc}func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, control-flow-sink, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)"
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
}