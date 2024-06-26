package me.alex_s168.uiua.ir

import blitz.collections.contents
import me.alex_s168.uiua.*

class Analysis(val block: IrBlock) {
    val removed = mutableListOf<IrInstr>()
    val added = mutableListOf<IrInstr>()

    fun origin(v: IrVar): IrInstr? =
        block.instrDeclFor(v)

    fun function(v: IrVar): IrBlock? =
        block.funDeclFor(v)?.second

    fun callers() =
        block.ref.values.filter { v ->
            v.instrs.any {
                it.instr is PushFnRefInstr && it.instr.fn == block.name
            }
        }

    fun isCalling(instr: IrInstr, idx: Int) =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prim.CALL,
                Prim.REDUCE,
                Prim.ROWS,
                Prim.TABLE,
                Prim.EACH -> idx == 0
                Prim.Comp.REPEAT -> idx == 2
                Prim.SWITCH -> idx == 1
                else -> false
            }
            else -> false
        }

    fun callerInstrs(): List<Pair<IrBlock, IrInstr>> {
        val res = mutableListOf<Pair<IrBlock, IrInstr>>()
        callers().forEach { block ->
            val ba = Analysis(block)
            block.instrs
                .filter { it.instr is PushFnRefInstr && it.instr.fn == block.name }
                .forEach { inst ->
                    val ref = inst.outs[0]
                    ba.trace(ref) { a, b, v ->
                        if (isCalling(b, b.args.indexOf(v))) {
                            res += a to b
                        }
                    }
                }
        }
        return res
    }

    fun trace(v: IrVar, fn: (block: IrBlock, instr: IrInstr, v: IrVar) -> Unit) {
        usages(v).forEach {
            if (it != null) {
                fn(block, it, v)
                it.outs.forEach {
                    trace(it, fn)
                }
            }
        }
        callerInstrs().forEach { (b, i) ->
            fn(b, i, v)
        }
    }

    fun isPrim(instr: IrInstr, kind: String? = null) =
        instr.instr is PrimitiveInstr && kind?.let { instr.instr.id == it } ?: true

    fun usages(v: IrVar) =
        block.instrs
            .filter { it.args.any { it.id == v.id } } +
        block.rets
            .filter { it.id == v.id }
            .map { null }

    fun unused(v: IrVar) =
        usages(v).isEmpty()

    fun rename(from: IrVar, to: IrVar) =
        block.updateVar(from, to)

    inline fun transform(
        onIn: List<IrInstr>,
        crossinline each: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar) -> Unit
    ) {
        val pre = block.instrs.toMutableList()
        val on = if (onIn === block.instrs) onIn.toList() else onIn
        on.forEach {
            var idx = block.instrs.indexOf(it)
            block.instrs.removeAt(idx)

            runCatching {
                each(it, { block.instrs.add(idx++, it) }, block::newVar)
            }.onFailure { err ->
                println("in transform on ${onIn.contents}:")
                block.instrs.clear()
                block.instrs.addAll(pre)
                println("(restored pre transform state; indecies in further error messages might be off)")
                throw err
            }
        }
        added.addAll(block.instrs - pre)
        removed.addAll(pre - block.instrs)
    }

    fun finish(dbgName: String) {
        println("pass $dbgName finished:")
        if (removed.isNotEmpty()) {
            println("  removed:")
            removed.forEach {
                println("    $it")
            }
        }
        if (added.isNotEmpty()) {
            println("  added:")
            added.forEach {
                println("    $it")
            }
        }
    }

    companion object {
        val argArrayUsing = mapOf(
            Prim.Comp.ARR_ALLOC to 0,
            Prim.Comp.ARR_STORE to 1,
            Prim.Comp.ARR_LOAD to 1,
            Prim.RESHAPE to 0,
        )

        val pervasive = arrayOf(
            Prim.ADD,
            Prim.SUB,
            Prim.MUL,
            Prim.DIV,
            Prim.LT,
            Prim.EQ,
            Prim.POW
        )
    }
}

inline fun List<IrInstr?>.allNN(cond: (IrInstr) -> Boolean) =
    all { it != null && cond(it) }

inline fun List<IrInstr?>.allPrim(cond: (String) -> Boolean) =
    allNN { it.instr is PrimitiveInstr && cond(it.instr.id) }

fun List<IrInstr>.allPrim(type: String) =
    allPrim { it == type }