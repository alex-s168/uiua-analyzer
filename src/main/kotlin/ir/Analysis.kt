package me.alex_s168.uiua.ir

import blitz.Either
import blitz.collections.caching
import blitz.collections.contents
import blitz.unreachable
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

    fun isEmpty(instr: IrInstr) =
        isPrim(instr, Prim.Comp.SINK)

    fun isLast(instr: IrInstr) =
        block.instrs
            .filterIndexed { index, _ -> index > block.instrs.indexOf(instr) }
            .all(::isEmpty)

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

    fun terminating(): Boolean =
        block.instrs.any { terminating(it) }

    fun terminating(instr: IrInstr) =
        isPrim(instr, Prim.Comp.PANIC) ||
        (getDefinetlyCalling(instr)?.all {
            function(it.get())?.let {
                Analysis(it).terminating()
            } ?: false
        } ?: false)

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

    fun getDefinetlyCalling(instr: IrInstr): List<VarRef>? =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prim.CALL -> listOf(VarRef.ofListElem(instr.args, 0))

                Prim.SWITCH -> origin(instr.args[1])!!.args
                    .let { li -> List(li.size) { i ->
                        VarRef.ofListElem(li, i) } }

                Prim.FILL -> instr.args
                    .let { li -> (0..<1).map {
                        VarRef.ofListElem(li, it) } }

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

    // WARNING: this takes up a lot of time from compilation
    fun callerInstrs(
        callis: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): Sequence<Pair<IrBlock, IrInstr>> =
        this.block.ref.values
            .asSequence()
            .flatMap { b ->
                val ba = Analysis(b)
                val bcall = callis(b)
                b.instrs.asSequence()
                    .filter { inst ->
                        inst.instr is PushFnRefInstr
                                && inst.instr.fn == this.block.name
                    }
                    .flatMap {
                        ba.trace(it.outs[0], bcall)
                            .filter { (_, i, v) ->
                                isCalling(i, i.args.indexOf(v)) }
                            .map { it.first to it.second }
                    }
            }
            .caching()

    // I hope it goes without saying that this is slower than slow
    fun deepOriginV2(
        v: IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): Either<Pair<IrBlock, IrInstr>, Double>? {
        origin(v)?.let {
            if (it.instr is NumImmInstr)
                return Either.ofB(it.instr.value)

            // required for type checker
            else if (isPrim(it, Prim.Comp.USE))
                return deepOriginV2(it.args[0], callerInstrsWrap)

            // required for type checker
            else if (isPrim(it, Prim.OVER))
                return when (it.outs.indexOf(v)) {
                    0 -> deepOriginV2(it.args[1], callerInstrsWrap)
                    1 -> deepOriginV2(it.args[0], callerInstrsWrap)
                    2 -> deepOriginV2(it.args[1], callerInstrsWrap)
                    else -> unreachable()
                }

            else
                return Either.ofA(this.block to it)
        }

        if (v in block.args) {
            val idx = block.args.indexOf(v)

            callerInstrsWrap(block).forEach { (callBlock, instr) ->
                val a = Analysis(callBlock)

                if (isPrim(instr, Prim.CALL)) {
                    a.deepOriginV2(instr.args[idx + 1], callerInstrsWrap)?.let { return it }
                }

                else if (isPrim(instr, Prim.SWITCH)) {
                    val ar = instr.args[idx + 3]

                    if (ar == instr.args[2]) { // identical to on param
                        val switchIdx = a.origin(instr.args[1])
                            ?.args
                            ?.map(a::function)
                            ?.indexOf(block)
                            ?: -1

                        if (switchIdx != -1) {
                            val from = a.origin(instr.args[0])!!.args[switchIdx]
                            a.deepOriginV2(from, callerInstrsWrap)?.let { return it }
                        }
                    }

                    a.deepOriginV2(ar, callerInstrsWrap)?.let { return it }
                }

                else if (isPrim(instr, Prim.Comp.REPEAT) && idx != 0) { // can't trace counter
                    a.deepOriginV2(instr.args[idx + 2], callerInstrsWrap)?.let { return it } // +3 -1  (-1 bc takes counter)
                }

                else if (isPrim(instr, Prim.FILL)) {
                    a.deepOriginV2(instr.args[idx + 2], callerInstrsWrap)?.let { return it }
                }
            }
        }

        return null
    }

    // TODO: GET RID OF THIS ASAP
    @Deprecated("does not handle all cases", replaceWith = ReplaceWith("deepOriginV2"))
    fun deepOrigin(v: IrVar): Pair<IrBlock, IrInstr>? =
        deepOriginV2(v)?.a

    // TODO: is trace useless?

    fun trace(v: IrVar, callerInstrs: Sequence<Pair<IrBlock, IrInstr>>): Sequence<Triple<IrBlock, IrInstr, IrVar>> {
        val first = usages(v)
            .filterNotNull()
            .flatMap {
                sequenceOf(Triple(block, it, v)) + it.outs
                    .asSequence()
                    .flatMap { trace(it, callerInstrs) }
            }
        return first + callerInstrs.map { (b, i) ->
            Triple(b, i, v)
        }
    }

    fun isPrim(instr: IrInstr, kind: String? = null) =
        instr.instr is PrimitiveInstr && kind?.let { instr.instr.id == it } ?: true

    fun usages(v: IrVar): Sequence<IrInstr?> =
        block.instrs
            .asSequence()
            .filter { it.args.any { it.id == v.id } } +
                block.rets
                    .asSequence()
                    .filter { it.id == v.id }
                    .map { null }

    fun recUsages(v: IrVar): Sequence<IrInstr> =
        usages(v)
            .filterNotNull()
            .flatMap { it.outs.asSequence().flatMap(::recUsages) + it }

    fun usagesAfter(v: IrVar, inst: IrInstr) =
        block.instrs.indexOf(inst).let { instIdx ->
            usages(v)
                .map(block.instrs::indexOf)
                .filter { it > instIdx }
                .map(block.instrs::get)
        }

    fun constNum(
        v: IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): Double? =
        deepOriginV2(v, callerInstrsWrap)?.b

    fun unused(v: IrVar) =
        usages(v).none()

    fun rename(from: IrVar, to: IrVar) =
        block.updateVar(from, to)

    inline fun <R> transform(
        onIn: List<IrInstr>,
        crossinline each: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar) -> R
    ): List<R> {
        val pre = block.instrs.toMutableList()
        val on = if (onIn === block.instrs) onIn.toList() else onIn
        val res = on.map {
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
            }.getOrThrow()
        }
        added.addAll(block.instrs - pre)
        removed.addAll(pre - block.instrs)
        return res
    }

    fun finish(dbgName: String) {
        if (removed.isNotEmpty() || added.isNotEmpty()) {
            println("pass $dbgName finished on block \"${block.name}\":")
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

    private fun recDependenciesImpl(block: List<IrInstr>, vars: MutableSet<IrVar>, instrs: MutableSet<IrInstr>) {
        allDependencies(block).forEach {
            vars += it
            origin(it)?.let {
                if (it !in instrs) {
                    recDependenciesImpl(listOf(it), vars, instrs)
                }
            }
        }
    }

    fun recDependencies(block: List<IrInstr>): Pair<Set<IrVar>, Set<IrInstr>> {
        val a = mutableSetOf<IrVar>()
        val b = mutableSetOf<IrInstr>()
        recDependenciesImpl(block, a, b)
        return a to b
    }

    fun constShape(
        arr: IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): List<Either<Int, Pair<IrBlock, IrVar>>>? {
        val (b, shape) = deepOriginV2(arr, callerInstrsWrap)?.a?.let { (b, v) ->
            // TODO: make work for arg arr and materialize
            if (Analysis(b).isPrim(v, Prim.Comp.ARR_ALLOC))
                b.instrDeclFor(v.args[0])
                    ?.args
                    ?.let { b to it }
            else null
        } ?: return null

        return shape.map {
            Analysis(b).deepOriginV2(it, callerInstrsWrap)
                ?.b?.toInt()?.let { Either.ofA(it) }
                ?: Either.ofB(b to it)
        }
    }

    fun isConstant(instr: IrInstr) =
        when (instr.instr) {
            is PushFnRefInstr,
            is NumImmInstr,
            is ArrImmInstr -> true

            else -> false
        }

    fun independentOfArrayData(instr: IrInstr) =
        when (instr.instr) {
            is PrimitiveInstr -> instr.instr.id in independentOfArrayData

            is PushFnRefInstr,
            is NumImmInstr,
            is ArrImmInstr -> true

            else -> false
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
            Prim.POW,
            Prim.MOD,
            Prim.SQRT,
            Prim.NEG,
            Prim.SIN,
            Prim.ASIN,
            Prim.FLOOR,
            Prim.CEIL,
            Prim.ROUND,
        )

        val independentOfArrayData = arrayOf(
            Prim.Comp.DIM,
            Prim.ADD,
            Prim.SUB,
            Prim.MUL,
            Prim.DIV,
            Prim.POW,
            Prim.MAX,
            Prim.LT,
            Prim.EQ,
        )

        fun blocksEqual(block1: IrBlock, block2: IrBlock): Boolean {
            if (block1.args.size != block2.args.size)
                return false

            val variables = block1.args
                .mapIndexed { index, irVar -> IrVar(id = index.toULong(), type = irVar.type) }

            val ib1 = block1.instrs.map{it}
            block1.args.zip(variables).forEach { (old, new) ->
                ib1.forEach { it.updateVar(old, new) }
            }

            val ib2 = block2.instrs.map{it}
            block2.args.zip(variables).forEach { (old, new) ->
                ib2.forEach { it.updateVar(old, new) }
            }

            if (ib1.size != ib2.size)
                return false

            return ib1.zip(ib2).all { (i1, i2) ->
                i1.args.contents == i2.args.contents &&
                i1.outs.contents == i2.outs.contents &&
                i1.instr == i2.instr
            }
        }
    }
}

inline fun Iterable<IrInstr?>.allNN(cond: (IrInstr) -> Boolean) =
    all { it != null && cond(it) }

inline fun Iterable<IrInstr?>.allPrim(cond: (String) -> Boolean) =
    allNN { it.instr is PrimitiveInstr && cond(it.instr.id) }

fun Iterable<IrInstr>.allPrim(type: String) =
    allPrim { it == type }