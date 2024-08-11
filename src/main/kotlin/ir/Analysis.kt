package me.alex_s168.uiua.ir

import blitz.collections.contents
import me.alex_s168.uiua.*

class Analysis(val block: IrBlock) {
    val removed = mutableListOf<IrInstr>()
    val added = mutableListOf<IrInstr>()

    fun variables() =
        block.instrs.flatMap { it.outs } + block.args

    fun origin(v: IrVar): IrInstr? =
        block.instrDeclFor(v)

    fun function(v: IrVar): IrBlock? =
        block.funDeclFor(v)?.second

    fun callers() =
        this.block.ref.values.filter { v ->
            v.instrs.any {
                it.instr is PushFnRefInstr && it.instr.fn == this.block.name
            }
        }

    fun terminating() =
        block.terminating()

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
                .filter { it.instr is PushFnRefInstr && it.instr.fn == this.block.name }
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

    fun deepOrigin(v: IrVar): Pair<IrBlock, IrInstr>? {
        origin(v)?.let { return this.block to it }

        if (v in block.args) {
            val idx = block.args.indexOf(v)

            callerInstrs().forEach { (block, instr) ->
                val a = Analysis(block)

                if (isPrim(instr, Prim.CALL)) {
                    return a.deepOrigin(instr.args[idx + 1])
                }

                if (isPrim(instr, Prim.SWITCH)) {
                    return a.deepOrigin(instr.args[idx + 3])
                }

                if (isPrim(instr, Prim.Comp.REPEAT)) {
                    return a.deepOrigin(instr.args[idx + 2]) // +3 -1  (-1 bc takes counter)
                }
            }
        }

        return null
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

    fun recUsages(v: IrVar): List<IrInstr> =
        usages(v).filterNotNull()
            .flatMap { it.outs.flatMap(::recUsages) + it }

    fun usagesAfter(v: IrVar, inst: IrInstr) =
        block.instrs.indexOf(inst).let { instIdx ->
            usages(v)
                .map(block.instrs::indexOf)
                .filter { it > instIdx }
                .map(block.instrs::get)
        }

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
        if (removed.isNotEmpty() || added.isNotEmpty()) {
            println("pass $dbgName finished:")
        }
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

    fun isPure(instr: IrInstr): Boolean =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prim.Comp.USE,
                Prim.Comp.ARR_MATERIALIZE,
                Prim.Comp.DIM,
                Prim.ADD,
                Prim.SUB,
                Prim.MUL,
                Prim.DIV,
                Prim.POW,
                Prim.MAX,
                Prim.LT,
                Prim.EQ,
                Prim.PRIMES,
                Prim.RANGE -> true
                else -> false
            }
            else -> true
        }

    fun idxRange(instrs: List<IrInstr>): IntRange? {
        val idx = instrs.map(block.instrs::indexOf)
        if (idx.isEmpty()) return null
        return idx.min() .. idx.max()
    }

    fun canMove(instrs: List<IrInstr>): Boolean {
        val range = idxRange(instrs)
            ?: return true

        val inRange = block.instrs.withIndex()
            .filterTo(mutableListOf()) { it.index in range }

        inRange.removeIf { isPure(it.value) }
        inRange.removeIf { it.value in instrs }
        return inRange.isEmpty()
    }

    fun fnRefs() =
        callers().flatMap { blk ->
            blk.instrs
                .filter { it.instr is PushFnRefInstr && it.instr.fn == this.block.name }
                .map { blk to it }
        }

    fun updateFnType() {
        fnRefs().forEach { (blk, instr) ->
            val old = instr.outs[0]
            val new = old.copy(type = this.block.type())
            blk.updateVar(old, new)
        }
    }

    fun allRelatedInstrs(variable: IrVar, after: IrInstr, dest: MutableList<IrInstr?> = mutableListOf()): MutableList<IrInstr?> {
        val afterIdx = this.block.instrs.indexOf(after)

        val li = recUsages(variable)
            .toMutableList()
        this.block.instrDeclFor(variable)
            ?.let {
                if (this.block.instrs.indexOf(it) > afterIdx) {
                    li += it
                }
            }
        li.removeIf { it in dest }
        li.removeIf { this.block.instrs.indexOf(it) <= afterIdx }
        dest.addAll(li)
        li.flatMap { it.args }.map {
            allRelatedInstrs(it, after, dest)
        }
        li.flatMap { it.outs }.map {
            allRelatedInstrs(it, after, dest)
        }
        return dest
    }

    fun dependentCodeBlockAsMovable(variable: IrVar, after: IrInstr): List<IrInstr> {
        if (variable in block.rets) return listOf()

        val block = allRelatedInstrs(variable, after)
        if (null in block) return listOf()
        val blocknn = block.filterNotNullTo(mutableListOf())
        blocknn.removeIf { this.isPrim(it, Prim.Comp.SINK) }

        if (!canMove(blocknn)) return listOf()

        return blocknn.sortedBy { this.block.instrs.indexOf(it) }
    }

    fun allDependencies(block: List<IrInstr>): List<IrVar> {
        val all = block.flatMap(IrInstr::args)
            .distinct()
            .toMutableList()
        all.removeIf { dep -> block.any { dep in it.outs } }
        return all
    }

    companion object {
        val argArrayUsing = mapOf(
            Prim.Comp.ARR_ALLOC to 0,
            Prim.Comp.ARR_STORE to 1,
            Prim.Comp.ARR_LOAD to 1,
            Prim.RESHAPE to 0,
        )

        val boundChecked = mapOf( // second pair: array idx, idx idx
            Prim.Comp.ARR_LOAD to (0 to 1),
            Prim.Comp.ARR_STORE to (0 to 1),
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