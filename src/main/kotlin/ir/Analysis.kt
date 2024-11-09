package me.alex_s168.uiua.ir

import blitz.*
import blitz.collections.RefVec
import blitz.collections.caching
import blitz.collections.contents
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.transform.constant

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

fun List<VarRef>.filterCertainlyCalling(srcBlock: IrBlock, what: BlockId) =
    filter { srcBlock.instrDeclFor(it.get())
        ?.let { it.instr is PushFnRefInstr && it.instr.fn == what } == true }

class Analysis(val block: IrBlock) {
    val removed = mutableListOf<IrInstr>()
    val added = mutableListOf<IrInstr>()

    fun isEmpty(instr: IrInstr) =
        isPrim(instr, Prims.Comp.SINK)

    fun isLast(instr: IrInstr) =
        block.instrs
            .filterIndexed { index, _ -> index > block.instrs.indexOf(instr) }
            .all(::isEmpty)

    fun variables() =
        block.instrs.asSequence().flatMap { it.outs } + block.args

    fun origin(v: IrVar): IrInstr? =
        block.instrDeclFor(v)

    fun function(
        v: IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): IrBlock? =
        deepOriginV2(v, callerInstrsWrap)
            ?.a?.second
            ?.instr?.cast<PushFnRefInstr>()
            ?.fn?.let(block.ref::get)

    fun callers() =
        this.block.ref.values
            .asSequence()
            .filter { v ->
                v.instrs.any {
                    it.instr is PushFnRefInstr && it.instr.fn == this.block.uid
                }
            }

    fun terminating(): Boolean =
        block.instrs.any { terminating(it) }

    fun terminating(instr: IrInstr) =
        isPrim(instr, Prims.Comp.PANIC) ||
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
                Prims.CALL,
                Prims.REDUCE,
                Prims.ROWS,
                Prims.TABLE,
                Prims.EACH -> 0
                Prims.Comp.REPEAT -> 2
                Prims.SWITCH -> 1
                Prims.FILL -> 1
                else -> null
            }
            else -> null
        }

    fun getDefinetlyCalling(instr: IrInstr): List<VarRef>? =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prims.CALL -> listOf(VarRef.ofListElem(instr.args, 0))

                Prims.SWITCH -> origin(instr.args[1])!!.args
                    .let { li -> List(li.size) { i ->
                        VarRef.ofListElem(li, i) } }

                Prims.FILL -> instr.args
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
                Prims.CALL,
                Prims.REDUCE,
                Prims.ROWS,
                Prims.TABLE,
                Prims.EACH -> listOf(VarRef.ofListElem(instr.args, 0))

                Prims.Comp.REPEAT -> listOf(VarRef.ofListElem(instr.args, 2))

                Prims.SWITCH -> origin(instr.args[1])!!.args
                    .let { li -> List(li.size) { i ->
                        VarRef.ofListElem(li, i) } }

                Prims.FILL -> instr.args
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
                                && inst.instr.fn == this.block.uid
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
            else if (isPrim(it, Prims.Comp.USE))
                return deepOriginV2(it.args[0], callerInstrsWrap)

            // required for type checker
            else if (isPrim(it, Prims.OVER))
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

                if (isPrim(instr, Prims.CALL)) {
                    a.deepOriginV2(instr.args[idx + 1], callerInstrsWrap)?.let { return it }
                }

                else if (isPrim(instr, Prims.SWITCH)) {
                    val ar = instr.args[idx + 3]

                    // TODO: ??????
                    /*
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
                    }*/

                    a.deepOriginV2(ar, callerInstrsWrap)?.let { return it }
                }

                else if (isPrim(instr, Prims.Comp.REPEAT) && idx != 0) { // can't trace counter
                    a.deepOriginV2(instr.arg(idx + 2), callerInstrsWrap)?.let { return it } // +3 -1  (-1 bc takes counter)
                }

                else if (isPrim(instr, Prims.FILL)) {
                    a.deepOriginV2(instr.arg(idx + 2), callerInstrsWrap)?.let { return it }
                }
            }
        }

        return null
    }

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

    fun isPrim(instr: IrInstr, kind: Prim? = null) =
        instr.instr is PrimitiveInstr && kind?.let { instr.instr.id == it } ?: true

    fun usages(v: IrVar): Sequence<IrInstr?> =
        block.instrs
            .asSequence()
            .filter { v in it.args }
            .let {
                if (v in block.rets) it + null else it
            }

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

    fun <R> modify(
        first: IrInstr,
        run: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar) -> R
    ): R {
        val pre = block.instrs.toMutableList()

        var idx = block.instrs.indexOf(first)

        val res = runCatching {
            run(first, { block.instrs.add(idx++, it) }, block::newVar)
        }.onFailure { err ->
            log("in transform starting at ${first}:")
            block.instrs.clear()
            block.instrs.addAll(pre)
            log("(restored pre transform state; indecies in further error messages might be off)")
            throw err
        }.getOrThrow()

        added.addAll(block.instrs - pre)
        removed.addAll(pre - block.instrs)
        return res
    }

    fun <R> transform(
        onIn: List<IrInstr>,
        each: IrInstr.(put: (IrInstr) -> Unit, newVar: () -> IrVar) -> R
    ): List<R> {
        val pre = block.instrs.toMutableList()
        val on = if (onIn === block.instrs) onIn.toList() else onIn
        val res = on.map {
            var idx = block.instrs.indexOf(it)
            block.instrs.removeAt(idx)

            runCatching {
                each(it, { block.instrs.add(idx++, it) }, block::newVar)
            }.onFailure { err ->
                log("in transform on ${onIn.contents}:")
                block.instrs.clear()
                block.instrs.addAll(pre)
                log("(restored pre transform state; indecies in further error messages might be off)")
                throw err
            }.getOrThrow()
        }
        added.addAll(block.instrs - pre)
        removed.addAll(pre - block.instrs)
        return res
    }

    fun finish(dbgName: String) {
        if (removed.isNotEmpty() || added.isNotEmpty()) {
            log("pass $dbgName finished on block \"${block.name}\":")
        }
        if (removed.isNotEmpty()) {
            log("  removed:")
            removed.forEach {
                log("    $it")
            }
        }
        if (added.isNotEmpty()) {
            log("  added:")
            added.forEach {
                log("    $it")
            }
        }
    }

    fun isPure(instr: IrInstr): Boolean =
        when (instr.instr) {
            is PrimitiveInstr -> when (instr.instr.id) {
                Prims.Comp.USE,
                Prims.Comp.DIM,
                Prims.ADD,
                Prims.SUB,
                Prims.MUL,
                Prims.DIV,
                Prims.POW,
                Prims.MAX,
                Prims.LT,
                Prims.EQ,
                Prims.PRIMES,
                Prims.RANGE -> true
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

        val inRange = block.instrs
            .filterIndexedTo(mutableListOf()) { idx,_ -> idx in range }

        inRange.removeIf(::isPure)
        inRange.removeAll(instrs)
        return inRange.isEmpty()
    }

    fun allRelatedInstrs(variable: IrVar, after: IrInstr, dest: RefVec<IrInstr> = RefVec()): RefVec<IrInstr> {
        val beginNew = dest.size
        val afterIdx = this.block.instrs.indexOf(after)
        fun filter(it: IrInstr) =
            it !in dest && this.block.instrs.indexOf(it) > afterIdx
        recUsages(variable)
            .filter(::filter)
            .toVec(dest)
        this.block.instrDeclFor(variable)
            ?.let { if (filter(it)) dest.pushBack(it) }
        repeat(dest.size - beginNew) { _idx ->
            val it = dest[_idx + beginNew]
            repeat(it.args.size) { idx ->
                allRelatedInstrs(it.args[idx], after, dest)
            }
            repeat(it.outs.size) { idx ->
                allRelatedInstrs(it.outs[idx], after, dest)
            }
        }
        return dest
    }

    fun dependentCodeBlockAsMovable(variable: IrVar, after: IrInstr): List<IrInstr> {
        if (variable in block.rets) return listOf()

        val block = allRelatedInstrs(variable, after)
            .fastToMutableList()
        block.removeIf { this.isPrim(it, Prims.Comp.SINK) }

        if (!canMove(block)) return listOf()

        return block.sortedBy { this.block.instrs.indexOf(it) }
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
            if (Analysis(b).isPrim(v, Prims.Comp.ARR_ALLOC))
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

    fun argArr(
        arr: IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): Either<List<Double>, List<IrVar>>? =
        deepOriginV2(arr, callerInstrsWrap)?.a?.second
            ?.let {
                if (it.instr is ArrImmInstr)
                    it.instr.values
                        .mapA { it.map(Int::toDouble) }
                        .flatten()
                        .let { Either.ofA(it) }
                else if (isPrim(it, Prims.Comp.ARG_ARR))
                    Either.ofB(it.args)
                else null
            }

    fun argArrAsConsts(
        arr: IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): List<Double>? =
        argArr(arr, callerInstrsWrap)
            ?.mapBOrNull { it.mapOrNull { deepOriginV2(it, callerInstrsWrap)?.b }?.fastToMutableList() }
            ?.flatten()

    fun argArrAsVars(
        arr: IrVar,
        put: (IrInstr) -> Unit,
        newVar: () -> IrVar,
        callerInstrsWrap: (IrBlock) -> Sequence<Pair<IrBlock, IrInstr>> = { Analysis(it).callerInstrs() }
    ): List<IrVar>? =
        argArr(arr, callerInstrsWrap)
            ?.mapA { it.map { constant(it, newVar = newVar, put = put) } }
            ?.flatten()

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
            Prims.Comp.ARR_ALLOC to 0,
            Prims.Comp.ARR_STORE to 1,
            Prims.Comp.ARR_LOAD to 1,
            Prims.RESHAPE to 0,
        )

        val forceMaterialize = mapOf(
            Prims.Comp.RESHAPE_VIEW to 0
        )

        val pervasive = arrayOf(
            Prims.ADD,
            Prims.SUB,
            Prims.MUL,
            Prims.DIV,
            Prims.LT,
            Prims.EQ,
            Prims.POW,
            Prims.MOD,
            Prims.SQRT,
            Prims.NEG,
            Prims.SIN,
            Prims.ASIN,
            Prims.FLOOR,
            Prims.CEIL,
            Prims.ROUND,
            Prims.MAX,
            Prims.MIN,
        )

        val independentOfArrayData = arrayOf(
            Prims.Comp.DIM,
            Prims.ADD,
            Prims.SUB,
            Prims.MUL,
            Prims.DIV,
            Prims.POW,
            Prims.MAX,
            Prims.LT,
            Prims.EQ,
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

inline fun Iterable<IrInstr?>.allPrim(cond: (Prim) -> Boolean) =
    allNN { it.instr is PrimitiveInstr && cond(it.instr.id) }

fun Iterable<IrInstr>.allPrim(type: Prim) =
    allPrim { it == type }
