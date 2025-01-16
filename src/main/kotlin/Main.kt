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

// TODO: EitherVec by adding PairVec
// TODO: fast contains on RefVec
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
        { block: IrBlock -> block.instrs.filter { it.instr !is CommentInstr }.size <= max }
}

object UnrollLoop {
    val none = { _: IrBlock, _: Int -> false }
    val all = { _: IrBlock, _: Int -> true }
    fun inlineCfg(i: (IrBlock) -> Boolean) =
        { block: IrBlock, _: Int -> i(block) }
    fun lteInlineCfg(maxLoopIters: Int, i: (IrBlock) -> Boolean) =
        { block: IrBlock, c: Int -> c <= maxLoopIters && i(block) }
    fun sumLte(max: Int) =
        { block: IrBlock, c: Int -> block.instrs.filter { it.instr !is CommentInstr }.size * c <= max }
}

var log = { str: String ->
    println(str)
}

val inlineConfig = Inline.lte(64)
val unfilledLoadBoundsCheck = false
val fullUnrollLoop = UnrollLoop.all // TODO: why sumLte no work
val boundsChecking = true // pick and unpick bounds checking
val mlirComments = true
val dontCareOpsBeforePanic = true
val additionalDebugInstrs = false // don't really work
val uacPrintSpans = true
val debugVerify = false

// TODO: (IMPORTANT) don't use scf.parallel if it can't safely be executed in parallel

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

fun <R> stage(name: String, fn: () -> R): R {
    print("$name: ")
    var result: R
    val ms = measureTimeMillis {
        result = fn()
    }
    println("$ms ms")
    return result
}

data class Stages(
    val first: Stage,
    val last: Stage,
) {
    init {
        val fromId = Stage.entries.indexOf(first)
        val toId = Stage.entries.indexOf(last)
        if (fromId >= toId)
            error("cannot compile in this order of stages")
    }

    companion object {
        private val splitAt = CharArray(26){'A'+it} + '_'
        val stages = listOf(
            "uiua", "uasm",
            "graph", "raw_uac",
            "lowered_uac", "opt_uac",
            "raw_mlir", "opt_mlir",
            "llvm",
            "obj", "exe"
        )

        init {
            assert(stages.size == Stage.entries.size)
        }

        fun parseStage(str: String): Stage =
           Stage.entries[stages.indexOf(str)]

        fun parse(arg: String): Stages {
            val spl = arg
                .split("->")
                .map(String::trim)

            val err = "expected compilation stage range in format: A->B"
            return when (spl.size) {
                2 -> Stages(
                    parseStage(spl[0]),
                    parseStage(spl[1]),
                )

                else -> error(err)
            }
        }
    }

    enum class Stage {
        Uiua,
        Uasm,
        Graph,
        RawUac,
        LoweredUac,
        OptUac,
        RawMlir,
        OptMlir,
        Llvm,
        Obj,
        Exe,
    }
}

data class Settings(
    val stages: Stages,
    val mainFn: String,
    val inUiua: String?,
    val inUasm: String,
    val logFile: String,
    val graphOutFile: String? = null,
    val genUac: String? = null,
    val loweredUac: String? = null,
    val optUac: String? = null,
    val genMlir: String,
    val optMlir: String,
    val genLlvm: String,
    val outObj: String,
    val outExe: String,
    val rtDir: String,
) {
    companion object {
        fun parse(defaultRt: String, argsIn: List<String>): Settings {
            val args = argsIn
                .associate { it.substringAfter("--").split("=").let { (a, b) -> a to b } }

            val stages = Stages.parse(args["pipeline"] ?: error("required argument \"pipeline\" not set"))

            val out = args["out"] ?: error("required argument \"out\" not set")

            fun temp(ext: String) = File.createTempFile("uiuac_", ext).absolutePath

            val inp = args["in"] ?: error("required argument \"in\" not set")

            return Settings(
                stages,
                args["main-fn"] ?: "fn",
                if (stages.first == Stages.Stage.Uiua) inp else null,
                if (stages.first == Stages.Stage.Uasm) inp else if (stages.last == Stages.Stage.Uasm) out else temp(".uasm"),
                args["log"] ?: ".log.txt",
                if (stages.last == Stages.Stage.Graph) out else null,
                if (stages.last == Stages.Stage.RawUac) out else null,
                if (stages.last == Stages.Stage.LoweredUac) out else null,
                if (stages.last == Stages.Stage.OptUac) out else null,
                if (stages.last == Stages.Stage.RawMlir) out else temp(".in.mlir"),
                if (stages.last == Stages.Stage.OptMlir) out else temp(".opt.mlir"),
                if (stages.last == Stages.Stage.Llvm) out else temp(".llc"),
                if (stages.last == Stages.Stage.Obj) out else temp(".o"),
                if (stages.last == Stages.Stage.Exe) out else temp(".exe"),
                args["rt"] ?: defaultRt
            )
        }
    }
}

fun main(args: Array<String>) {
    if (args.isEmpty() || arrayOf("-h", "--h", "-help", "--help").any { it in args }) {
        println("""
Array Language Compiler

Example:
    uiuac --pipeline=uiua->exe --in="my_uiua_code.ua" --out="my_exe.exe"

Arguments:
    --pipeline=[fmt]->[fmt]  specify from which file format to compile to which file format. available:
        uiua        (uiua source code)
        uasm        (uiua assembly)
        graph       (DOT graph of the syntax tree)
        raw_uac     (uiuac IR before any passes)
        lowered_uac (uiuac IR after lowering high-level primives)
        opt_uac     (uiuac IR after optimizations)
        raw_mlir    (MLIR after conversion of uiuac IR to MLIR)
        opt_mlir    (MLIR after mlir-opt)
        llvm        (LLVM IR generated by mlir-translate)
        obj         (Object file (ELF, COFF, ...))
        exe         (Executable file (ELF, PE, ...))

    --in=[path]     input file path

    --out=[path]    output file path

    --main-fn=[str] which function to compile (default: "fn")
                    setting this does not work if the pipeline ends with "exe"

    --log=[path]    the path of the log file (default: ".log.txt")

    --rt=[path]     path to the "rt" directory in the uiuac git repository
        """)
        return
    }

    compile(Settings.parse("rt", args.toList()))
}

@JvmName("generic_1")
private fun Pass<Unit>.generic(): Pass<(IrBlock) -> Unit> =
    Pass(name) { b, _ -> internalRun(b, Unit) }

@JvmName("generic_2")
private fun GlobalPass<Unit>.generic(): GlobalPass<(IrBlock) -> Unit> =
    GlobalPass(name) { b, _ -> internalRun(b, Unit) }

@JvmName("generic_3")
private fun Pass<(IrBlock) -> Unit>.generic() =
    this

@JvmName("generic_4")
private fun GlobalPass<(IrBlock) -> Unit>.generic() =
    this

private fun <A> GlobalPass<A>.setArg(v: A): GlobalPass<Unit> =
    GlobalPass(name) { b, _ -> this@setArg.internalRun(b, v) }

private fun lowerPasses(exported: List<BlockId>) = listOf(
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

    inlineCUse.generic(),
    fixArgArrays.generic(),
    argArrLoad.generic(),
    inlineCUse.generic(),

    lifetimes.generic(),
    remClone.generic(),

    lowerClone.generic(),
    lowerShape.generic(),
    lowerLen.generic(),
    // boundsChecking.generic(),  // TODO only for pick
    evalDim.generic(),
    remUnused.generic(),

    lowerFill.generic(),
    fixFnTypes.generic(),
)

private fun optPasses(exported: List<BlockId>) = listOf(
    verifyBlock.generic(),

    /*funcInline.generic(),
    funcInline.generic(),
    inlineCUse.generic(),
    fixArgArrays.generic(),
    evalDim.generic(),
    lowerArrCopy.generic(),
    evalDim.generic(),
    inlineCUse.generic(),
    inlineCUse.generic(),
    argArrLoad.generic(),
    inlineCUse.generic(),
    unrollLoop.generic(),
    inlineCUse.generic(),
    argArrLoad.generic(),
    inlineCUse.generic(),*/

    loadStore.generic(),
    argArrLoad.generic(),
    inlineCUse.generic(),

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
    loadStore.generic(),
    inlineCUse.generic(),
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
    loadStore.generic(),
    inlineCUse.generic(),
    loadStore.generic(),
    inlineCUse.generic(),
)

private fun preEmitPasses(exported: List<BlockId>) = listOf(
    argArrMat.generic(),
)

fun compile(cfg: Settings) {
    if (cfg.stages.last == Stages.Stage.Exe) {
        if (!File(cfg.rtDir).exists()) {
            println("set argument \"rt\" to point to a correct path (see \"--help\")")
            exitProcess(1)
        }

        if (!File(cfg.rtDir + "/build/rt_part0.a").exists()) {
            println("You did not build the uiuac runtime! See the README.md in the uiuac git repo")
            exitProcess(1)
        }
    }

    File(cfg.logFile).bufferedWriter().use { logwriter ->
        log = ConcurrentLogger(logwriter)::log

        fun run(cmd: List<String>) {
            println("\$ ${cmd.joinToString(" ")}")
            if (Runtime.getRuntime().exec(cmd.toTypedArray()).waitFor() == 0)
                return
            println("could not run command")
            exitProcess(1)
        }

        when (cfg.stages.first) {
            Stages.Stage.Uiua -> stage("parse uiua") {
                run(listOf("uiua", "build", cfg.inUiua!!, "-o", cfg.inUasm))
            }

            Stages.Stage.Uasm -> {}

            else -> error("using anything but a .ua or .uasm file as input is currently not suppported")
        }

        if (cfg.stages.last == Stages.Stage.Uasm)
            return@use

        val (assembly, blocks) = stage("generating AST") {
            val test = File(cfg.inUasm).readText()
            val assembly = Assembly.parse(test)

            val astNodes = mutableListOf<ASTRoot>()
            val blocks = assembly.functions.toIr(astNodes)

            cfg.graphOutFile?.let(::File)?.writer()?.use { w ->
                w.append(astNodes.genGraph())
            }

            assembly to blocks
        }

        if (cfg.stages.last == Stages.Stage.Graph)
            return@use

        val exported = stage("typechecking") {
            val old = blocks.keys.toList()
            val exported = listOf(
                blocks
                    .find(cfg.mainFn)!!
                    .expandFor(listOf(), blocks::putBlock)
            )
            old.forEach(blocks::remove)
        
            exported
        }

        exported.forEach { blocks[it]!!.private = false }

        cfg.genUac?.let(::File)?.writer()?.use { w ->
            blocks.values.forEach {
                w.append(it.toString())
                w.append("\n\n")
            }
        }

        if (cfg.stages.last == Stages.Stage.RawUac)
            return@use

        val res = runCatching {
            stage("lowering primitives") {
                lowerPasses(exported).apply(blocks)
            }
            cfg.loweredUac?.let(::File)?.printWriter()?.use { file ->
                blocks.values.forEach {
                    file.println(it)
                    file.println("\n\n")
                }
            }

            if (cfg.stages.last == Stages.Stage.LoweredUac)
                return@use

            stage("optimizing") {
                optPasses(exported).apply(blocks)
            }
            cfg.optUac?.let(::File)?.printWriter()?.use { file ->
                blocks.values.forEach {
                    file.println(it)
                    file.println("\n\n")
                }
            }

            if (cfg.stages.last == Stages.Stage.OptUac)
                return@use

            preEmitPasses(exported).apply(blocks)
        }
        res.onFailure {
            log("================================================================================")
            log("in apply pass pipeline")
            throw it
        }
        val compile = blocks.values.toSet()

        log("emitting MLIR...")

        File(cfg.genMlir).printWriter().use { file ->
            file.println(loadRes("runtime.mlir")!!)
            file.println("\n")

            stage("emitting MLIR") {
                compile.forEach { block ->
                    file.println(block.emitMLIRFinalize(block.emitMLIR { loc ->
                        listOf(assembly.spans
                            .gather(loc.uasmSpanIdc)
                            .joinToString()
                            .let { "// source: $it" })
                    }))
                }
                Unit
            }
        }

        if (cfg.stages.last == Stages.Stage.RawMlir)
            return@use

        val mlirOpt = "mlir-opt"
        val mlirTranslate = "mlir-translate"
        val clang = "clang"

        val enableBufferDealloc = false // TODO

        val bufferDealloc =
            if (enableBufferDealloc) "ownership-based-buffer-deallocation, buffer-deallocation-simplification, " else ""
        val mlirPipeline =
            if (cfg.stages.last != Stages.Stage.OptMlir) "-pass-pipeline=builtin.module(func(cse, canonicalize), sccp, sroa, inline, symbol-dce, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), ${bufferDealloc}func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, control-flow-sink, canonicalize, memref-expand, expand-strided-metadata, lower-affine, math-uplift-to-fma, finalize-memref-to-llvm, convert-scf-to-cf, convert-to-llvm, reconcile-unrealized-casts, canonicalize)"
            else "-pass-pipeline=builtin.module(func(cse, canonicalize), sccp, sroa, inline, symbol-dce, canonicalize, loop-invariant-code-motion, control-flow-sink, loop-invariant-subset-hoisting, control-flow-sink, scf-forall-to-for, canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), ${bufferDealloc}func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=512}), convert-bufferization-to-memref, control-flow-sink, canonicalize)"
        val mlirOptFlags = listOf(mlirPipeline)
        val mlirTranslateFlags = listOf("--mlir-to-llvmir")
        val clangFlags = listOf("-x", "ir")

        run(listOf(mlirOpt, "-o", cfg.optMlir, cfg.genMlir) + mlirOptFlags)
        if (cfg.stages.last == Stages.Stage.OptMlir)
            return@use

        run(listOf(mlirTranslate, "-o", cfg.genLlvm, cfg.optMlir) + mlirTranslateFlags)
        if (cfg.stages.last == Stages.Stage.Llvm)
            return@use

        run(listOf(clang, "-c", "-O3", "-march=native", "-o", cfg.outObj) + clangFlags + cfg.genLlvm)
        if (cfg.stages.last == Stages.Stage.Obj)
            return@use

        run(listOf(clang, cfg.outObj, cfg.rtDir + "/build/rt_part0.a", "-o", cfg.outExe))

        println("Generated ${cfg.outExe}")
    }
}
