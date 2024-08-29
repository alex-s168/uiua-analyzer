package me.alex_s168.uiua.ir

import blitz.Either
import blitz.collections.contents
import me.alex_s168.uiua.*

interface VarRef {
    fun get(): IrVar
    fun replace(with: IrVar)

    companion object {
        fun ofListElem(list: MutableList<IrVar>, index: Int) =
            object : VarRef {
                override fun get(): IrVar =
                    list[index]

                override fun replace(with: IrVar) {
                    list[index] = with
                }
            }
    }
}

fun List<VarRef>.filterCertainlyCalling(srcBlock: IrBlock, what: String) =
    filter { srcBlock.instrDeclFor(it.get())
        ?.let { it.instr is PushFnRefInstr && it.instr.fn == what } == true }

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

    // TODO: stop using this and use getDeepCalling instead
    @Deprecated(
        message = "stupid",
        replaceWith = ReplaceWith("getDeepCalling"),
    )
    fun getCalling(instr: IrInstr): Int? =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prim.CALL,
                Prim.REDUCE,
                Prim.ROWS,
                Prim.TABLE,
                Prim.EACH -> 0
                Prim.Comp.REPEAT -> 2
                Prim.SWITCH -> 1
                Prim.FILL -> 1
                else -> null
            }
            else -> null
        }

    /** returns list of variables of fnrefs that might get called by given instr */
    fun getDeepCalling(instr: IrInstr): List<VarRef> =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prim.CALL,
                Prim.REDUCE,
                Prim.ROWS,
                Prim.TABLE,
                Prim.EACH -> listOf(VarRef.ofListElem(instr.args, 0))

                Prim.Comp.REPEAT -> listOf(VarRef.ofListElem(instr.args, 2))

                Prim.SWITCH -> origin(instr.args[1])!!.args
                    .let { li -> List(li.size) { i ->
                        VarRef.ofListElem(li, i) } }

                Prim.FILL -> instr.args
                    .let { li -> (0..<1).map {
                        VarRef.ofListElem(li, it) } }

                else -> emptyList()
            }
            else -> emptyList()
        }

    fun isCalling(instr: IrInstr, idx: Int) =
        getCalling(instr) == idx

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
                    a.deepOrigin(instr.args[idx + 1])?.let { return it }
                }

                if (isPrim(instr, Prim.SWITCH)) {
                    a.deepOrigin(instr.args[idx + 3])?.let { return it }
                }

                if (isPrim(instr, Prim.Comp.REPEAT)) {
                    a.deepOrigin(instr.args[idx + 2])?.let { return it } // +3 -1  (-1 bc takes counter)
                }

                if (isPrim(instr, Prim.FILL)) {
                    a.deepOrigin(instr.args[idx + 2])?.let { return it }
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

    fun constNum(v: IrVar) =
        (origin(v)?.instr as? NumImmInstr)?.value

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

    fun constShape(arr: IrVar): List<Either<Int, Pair<IrBlock, IrVar>>>? {
        val (b, shape) = deepOrigin(arr)?.let { (b, v) ->
            if (Analysis(b).isPrim(v, Prim.Comp.ARR_ALLOC))
                b to v
            else null
        }?.let { (b, a) -> b.instrDeclFor(a.args[0])
            ?.args
            ?.let { b to it }
        } ?: return null

        return shape.map {
            Analysis(b).deepOrigin(it)
                ?.let { (_, i) -> if (i.instr is NumImmInstr) Either.ofA(i.instr.value.toInt()) else null }
                ?: Either.ofB(b to it)
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