package me.alex_s168.uiua.ir

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.transform.comment


inline fun IrBlock.lowerPass(
    name: String,
    filter: (IrInstr) -> Boolean,
    crossinline block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) {
    val a = Analysis(this)
    a.transform(instrs.filter(filter)) { put, newVar ->
        put(comment("+++ $this"))
        block(this, put, newVar, a)
        put(comment("--- $this"))
    }
    val locs = a.removed.mapNotNull {
        if (it.instr is PrimitiveInstr) it.instr.loc?.index
        else null
    }.flatten()
    a.added.forEach {
        if (it.instr is PrimitiveInstr && it.instr.loc == null) {
            it.instr.loc = SpanRef(locs)
        }
    }
    a.finish(name)
}

fun IrBlock.lowerPrimPass(
    name: String,
    primitive: Prim,
    block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = lowerPass(name, { it.instr is PrimitiveInstr && it.instr.id == primitive }, block)

sealed interface AnyPass {
    val name: String
}

data class GlobalPass<A>(
    override val name: String,
    val internalRun: (MutableMap<BlockId, IrBlock>, A) -> Unit
): AnyPass

fun <A> GlobalPass<A>.run(map: MutableMap<BlockId, IrBlock>, arg: A) {
    kotlin.runCatching {
        internalRun(map, arg)
    }.onFailure {
        map.values.forEach {
            log(it.toString())
            log("")
        }
        log("in global pass $name with arg $arg")
        throw it
    }
}

data class Pass<A>(
    override val name: String,
    /** each block -> own thread */
    val parallel: Boolean = true,
    /** deep copy each block when parallel */
    val parallelDeepCopyBlocks: Boolean = true,
    val internalRun: (IrBlock, A) -> Unit,
): AnyPass

fun <A> Pass<A>.withoutParallel() =
    Pass(name, false, false, internalRun)

fun <A> Pass<A>.parallelWithoutDeepCopy() =
    Pass(name, true, false, internalRun)


fun <A> Pass<A>.run(block: IrBlock, arg: A) {
    kotlin.runCatching {
        internalRun(block, arg)
    }.onFailure {
        log(block.toString())
        log("in pass $name on block(${block.uid}) with arg $arg")
        throw it
    }
}

fun Pass<Unit>.run(block: IrBlock) =
    run(block, Unit)

fun <T> List<Pass<T>>.merge(name: String) =
    Pass<T>(name) { block, a ->
        forEach {
            it.run(block, a)
        }
    }

fun <A> analysisPass(name: String, run: (IrBlock, Analysis, A) -> Unit) =
    Pass<A>(name) { it, arg ->
        val a = Analysis(it)
        run(it, a, arg)
        a.finish(name)
    }

fun analysisPass(name: String, run: (IrBlock, Analysis) -> Unit) =
    Pass<Unit>(name) { it, _ ->
        val a = Analysis(it)
        run(it, a)
        a.finish(name)
    }

fun genericEachInstrPass(name: String, run: IrInstr.(IrBlock, Analysis) -> Unit) =
    Pass<Unit>(name) { blk, _ ->
        val a = Analysis(blk)
        blk.instrs.toList().forEach { it.run(blk, a) }
        a.finish(name)
    }

fun genericEachPrimPass(name: String, prim: Prim, run: IrInstr.(IrBlock, Analysis) -> Unit) =
    Pass<Unit>(name) { blk, _ ->
        val a = Analysis(blk)
        blk.instrs.toList().forEach {
            if (a.isPrim(it, prim))
                it.run(blk, a)
        }
        a.finish(name)
    }

fun lowerPrimPass(
    primitive: Prim,
    block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = Pass<Unit>("lower ${Prims.all[primitive]}") { it, _ -> it.lowerPrimPass("lower ${Prims.all[primitive]}", primitive, block) }

inline fun <A> lowerPrimPass(
    primitive: Prim,
    crossinline block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis, arg: A) -> Unit
) = Pass<A>("lower ${Prims.all[primitive]}") { it, arg ->
    it.lowerPrimPass("lower ${Prims.all[primitive]}", primitive) { put, newVar, a ->
        block(put, newVar, a, arg)
    }
}

fun lowerPass(
    name: String,
    filter: (IrInstr) -> Boolean,
    block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = Pass<Unit>(name) { it, _ -> it.lowerPass(name, filter, block) }

inline fun <A> lowerPass(
    name: String,
    noinline filter: (IrInstr) -> Boolean,
    crossinline block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis, arg: A) -> Unit
) = Pass<A>(name) { it, arg ->
    it.lowerPass(name, filter) { put, newVar, a ->
        block(put, newVar, a, arg)
    }
}

inline fun optAwayPass(
    name: String,
    crossinline filter: (IrInstr) -> Boolean,
    crossinline rmIf: IrInstr.(a: Analysis) -> Boolean,
    crossinline rm: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = Pass<Unit>(name) { it, _ ->
    val a = Analysis(it)
    val on = it.instrs
        .filter(filter)
        .filter { rmIf(it, a) }
    a.transform(on) { put, newVar -> rm(this, put, newVar, a) }
    a.finish(name)
}

inline fun optAwayPass(
    name: String,
    crossinline filter: (IrInstr) -> Boolean,
    crossinline rmIf: IrInstr.(a: Analysis) -> Boolean
) = optAwayPass(name, filter, rmIf) { put, newVar, a -> }

inline fun optAwayPass(
    name: String,
    primitive: Prim,
    crossinline rmIf: IrInstr.(a: Analysis) -> Boolean,
    crossinline rm: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = optAwayPass(name, { it.instr is PrimitiveInstr && it.instr.id == primitive }, rmIf, rm)

inline fun optAwayPass(
    name: String,
    primitive: Prim,
    crossinline rmIf: IrInstr.(a: Analysis) -> Boolean
) = optAwayPass(name, primitive, rmIf) { put, newVar, a -> }

inline fun modifyPass(
    name: String,
    crossinline filter: (IrInstr) -> Boolean,
    crossinline modIf: IrInstr.(a: Analysis) -> Boolean,
    crossinline mod: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = optAwayPass(name, filter, modIf) { put, newVar, a ->
    mod(put, newVar, a)
    put(this)
}

inline fun modifyPass(
    name: String,
    primitive: Prim,
    crossinline modIf: IrInstr.(a: Analysis) -> Boolean,
    crossinline mod: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = modifyPass(name, { it.instr is PrimitiveInstr && it.instr.id == primitive }, modIf, mod)

inline fun <A> withPassArg(
    name: String,
    crossinline inner: (A) -> Pass<Unit>
): Pass<A> =
    Pass(name) { it, a ->
        inner(a).run(it)
    }

fun <T> passPipeline(of: List<Pass<T>>): (IrBlock, T) -> Unit =
    { block, arg ->
        of.forEach {
            it.run(block, arg)
        }
    }