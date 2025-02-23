package me.alex_s168.uiua.ir

import blitz.collections.contents
import me.alex_s168.uiua.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

private var nextBlockId = AtomicInteger(0)
private var nextGVarId = AtomicLong(0)

enum class Lifetime {
    LOCAL,
    GLOBAL, // if the value gets stored in something that leaves the scope
    ;

    override fun toString(): String =
        when (this) {
            LOCAL -> "local"
            GLOBAL -> "global"
        }
}

data class IrBlock(
    var name: String,
    var ref: Map<BlockId, IrBlock>,
    var instrs: MutableList<IrInstr> = mutableListOf(),
    var flags: MutableList<String> = mutableListOf(),
    var args: MutableList<IrVar> = mutableListOf(),
    var rets: MutableList<IrVar> = mutableListOf(),
    var fillArg: IrVar? = null,
    var private: Boolean = true,
) {
    var uid: BlockId = nextBlockId.getAndIncrement()

    val lifetimes = mutableMapOf<IrVar, Lifetime>()

    fun loadFrom(other: IrBlock) {
        name = other.name
        ref = other.ref
        instrs = other.instrs
        flags = other.flags
        args = other.args
        rets = other.rets
        fillArg = other.fillArg
        private = other.private
        uid = other.uid
    }

    fun shouldInline(): Boolean =
        inlineConfig(this)

    fun newVar(): IrVar =
        IrVar(Types.tbd, nextGVarId.getAndIncrement().toULong())

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        rets.updateVar(old, new)
        instrs.forEach {
            it.updateVar(old, new)
            if (it.instr is PrimitiveInstr && it.instr.id == Prims.Comp.ARG_ARR && new in it.args && it.outs[0] != new) {
                updateVar(it.outs[0], it.outs[0].copy(type = Types.array(it.args[0].type, it.args.size)))
            }
        }
        if (fillArg == old)
            fillArg = new
    }

    fun instrDeclFor(variable: IrVar): IrInstr? =
        instrs.find { variable in it.outs }

    fun funDeclFor(v: IrVar): IrBlock? {
        val fn = instrDeclFor(v)?.instr as? PushFnRefInstr
        return fn?.let { a ->
            ref[a.fn]
        }
    }

    fun varUsed(variable: IrVar): Boolean =
        if (variable in rets) true
        else instrs.any { variable in it.args }

    fun deepCopyNoNewVar() =
        IrBlock(
            name,
            ref,
            instrs.mapTo(mutableListOf()) { it.deepCopy() },
            flags.toMutableList(),
            args.toMutableList(),
            rets.toMutableList(),
            fillArg,
            private
        )

    fun deepCopy(): IrBlock {
        val a = args.mapTo(mutableListOf()) { newVar().copy(type = it.type) }
        val r = rets.mapTo(mutableListOf()) { newVar().copy(type = it.type) }
        val d = inlinableCopy(
            a.toList(),
            r.toList(),
            fillArg
        )
        d.args = a
        d.rets = r
        return d
    }

    fun inlinableCopy(cArgs: List<IrVar>, cRets: List<IrVar>, fill: IrVar? = null, updateVar: (IrBlock, IrVar, IrVar) -> Unit = IrBlock::updateVar): IrBlock {
        val new = IrBlock(
            anonFnName(),
            ref,
            instrs.mapTo(mutableListOf()) { it.deepCopy() },
            flags.toMutableList(),
            fillArg = fillArg,
            private = this.private
        )

        val olds = new.instrs.flatMap { it.outs }.toSet()
        olds.forEach {
            val n = (if (it in rets)
                cRets.getOrNull(rets.indexOf(it))
            else null) ?: new.newVar().copy(type = it.type)
            updateVar(new, it, n)
        }

        new.fillArg?.let { new.instrs.forEach { it.args.updateVar(new.fillArg!!, fill!!) } }

        new.fillArg = fill

        args.zip(cArgs).forEach { (a, b) ->
            require(a.type.similar(b.type)) {
                "inlining arg types do not match:\n  call with ${cArgs.map { it.type }.contents}\n  fn with ${args.map { it.type }.contents}"
            }
            if (a != b)
                updateVar(new, a, b)
        }

        return new
    }

    fun expandFor(
        inTypes: List<Type>,
        putFn: (IrBlock) -> Unit,
        fillType: Type? = null
    ): BlockId {
        require(inTypes.size == args.size) {
            "${inTypes.size} vs ${args.size}"
        }

        return runCatching {
            val newName = "${name}_\$_${fillType?.toString() ?: ""}_${inTypes.joinToString(separator = "_")}"

            val new = IrBlock(
                newName,
                ref,
                instrs.mapTo(mutableListOf()) { it.deepCopy() },
                flags.toMutableList(),
                args.toMutableList(),
                rets.toMutableList(),
            )

            new.args
                .zip(inTypes)
                .forEach { (v, t) ->
                    require(t != Types.tbd)
                    new.updateVar(v, v.copy(type = t))
                }

            new.instrs.toList().forEach {
                it.inferTypes(new, putFn, fillType)
            }

            fillType?.let {
                new.fillArg = new.newVar().copy(type = it)
            }

            putFn(new)

            new.uid
        }.getOrElse {
            log(this.toString())
            error("While trying to expand function \"$name\" as func${fillType?.let { "[$it]" } ?: ""}[${inTypes.contents}][?]:\n${it.stackTraceToString()}")
        }
    }

    override fun toString(): String {
        val res = StringBuilder()

        flags.forEach {
            res.append('@')
            res.append(it)
            res.append('\n')
        }

        res.append("block($uid) ")
        res.append(name)
        fillArg?.let {
            res.append(" fill ")
            res.append(it)
        }
        res.append(" (")
        args.forEachIndexed { index, irVar ->
            if (index > 0)
                res.append(", ")
            lifetimes[irVar]?.let {
                res.append(it)
                res.append(" ")
            }
            res.append(irVar)
        }
        res.append(")\n")

        val idxSize = (instrs.size - 1).toString().length
        instrs.forEachIndexed { index, it ->
            res.append("${index.toString().padStart(idxSize)}| ")
            res.append(it.toString())
            res.append('\n')
        }

        res.append("return ")
        rets.forEachIndexed { index, irVar ->
            if (index > 0)
                res.append(", ")
            res.append(irVar)
        }

        return res.toString()
    }

    fun type(): FnType =
        Types.func(
            args.map { it.type },
            rets.map { it.type },
            fillArg?.type
        )

    override fun hashCode(): Int =
        uid.hashCode()
}