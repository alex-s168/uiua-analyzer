package me.alex_s168.uiua.ir

import blitz.collections.contents
import me.alex_s168.uiua.*

private var nextBlockId = 0
private var nextGVarId: ULong = 0u

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
    val name: String,
    val ref: Map<String, IrBlock>,
    var instrs: MutableList<IrInstr> = mutableListOf(),
    var flags: MutableList<String> = mutableListOf(),
    var args: MutableList<IrVar> = mutableListOf(),
    var rets: MutableList<IrVar> = mutableListOf(),
    var fillArg: IrVar? = null,
    var private: Boolean = true,
) {
    val uid = nextBlockId ++

    val lifetimes = mutableMapOf<IrVar, Lifetime>()

    fun shouldInline(): Boolean =
        inlineConfig(this)

    fun newVar(): IrVar =
        IrVar(Types.tbd, nextGVarId ++)

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        rets.updateVar(old, new)
        instrs.forEach {
            it.updateVar(old, new)
        }
        if (fillArg == old)
            fillArg = new
    }

    fun instrDeclFor(variable: IrVar): IrInstr? =
        instrs.find { variable in it.outs }

    fun funDeclFor(v: IrVar): Pair<String, IrBlock>? {
        val fn = instrDeclFor(v)?.instr as? PushFnRefInstr
        return fn?.let { a ->
            ref[a.fn]?.let { a.fn to it }
        }
    }

    fun terminating(): Boolean =
        instrs.any { it.instr is PrimitiveInstr && it.instr.id == Prim.Comp.PANIC }

    fun varUsed(variable: IrVar): Boolean =
        if (variable in rets) true
        else instrs.any { variable in it.args }

    fun inlinableCopy(cArgs: List<IrVar>, cRets: List<IrVar>, fill: IrVar? = null): IrBlock {
        val new = IrBlock(
            name,
            ref,
            instrs.mapTo(mutableListOf()) { it.deepCopy() },
            flags.toMutableList(),
            fillArg = fillArg,
        )

        val olds = new.instrs.flatMap { it.outs }.toSet()
        olds.forEach {
            val n = (if (it in rets)
                cRets.getOrNull(rets.indexOf(it))
            else null) ?: new.newVar().copy(type = it.type)
            new.updateVar(it, n)
        }

        new.fillArg?.let { new.instrs.forEach { it.args.updateVar(new.fillArg!!, fill!!) } }

        new.fillArg = fill

        args.zip(cArgs).forEach { (a, b) ->
            require(a.type == b.type)
            new.updateVar(a, b)
        }

        return new
    }

    fun expandFor(
        inTypes: List<Type>,
        putFn: (IrBlock) -> Unit,
        fillType: Type? = null
    ): String {
        require(inTypes.size == args.size)

        return runCatching {
            val newName = "${name}_\$_${fillType?.toString() ?: ""}_${inTypes.joinToString(separator = "_")}"
            if (ref[newName] != null) return newName

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

            newName
        }.getOrElse {
            error("While trying to expand function \"$name\" as func${fillType?.let { "[$it]" } ?: ""}[${inTypes.contents}][?]: $it")
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