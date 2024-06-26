package me.alex_s168.uiua.ir

import me.alex_s168.uiua.PrimitiveInstr
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
    a.finish(name)
}

fun IrBlock.lowerPrimPass(
    name: String,
    primitive: String,
    block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = lowerPass(name, { it.instr is PrimitiveInstr && it.instr.id == primitive }, block)

data class Pass<A>(
    val name: String,
    val internalRun: (IrBlock, A) -> Unit,
)

fun <A> Pass<A>.run(block: IrBlock, arg: A) {
    kotlin.runCatching {
        internalRun(block, arg)
    }.onFailure {
        println("in pass $name on block(${block.uid}) with arg $arg")
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

fun lowerPrimPass(
    primitive: String,
    block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = Pass<Unit>("lower $primitive") { it, _ -> it.lowerPrimPass("lower $primitive", primitive, block) }

inline fun <A> lowerPrimPass(
    primitive: String,
    crossinline block: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis, arg: A) -> Unit
) = Pass<A>("lower $primitive") { it, arg ->
    it.lowerPrimPass("lower $primitive", primitive) { put, newVar, a ->
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
    primitive: String,
    crossinline rmIf: IrInstr.(a: Analysis) -> Boolean,
    crossinline rm: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar, a: Analysis) -> Unit
) = optAwayPass(name, { it.instr is PrimitiveInstr && it.instr.id == primitive }, rmIf, rm)

inline fun optAwayPass(
    name: String,
    primitive: String,
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
    primitive: String,
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