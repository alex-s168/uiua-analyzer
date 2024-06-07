package me.alex_s168.uiua.ir

import me.alex_s168.uiua.PushFnRefInstr
import me.alex_s168.uiua.Type
import me.alex_s168.uiua.Types
import kotlin.math.max

data class IrBlock(
    val name: String,
    val ref: (String) -> IrBlock?,
    var instrs: MutableList<IrInstr> = mutableListOf(),
    var flags: MutableList<String> = mutableListOf(),
    var args: MutableList<IrVar> = mutableListOf(),
    var rets: MutableList<IrVar> = mutableListOf(),
    var fillArg: IrVar? = null,
    var private: Boolean = true,
) {
    var nextVar: ULong = 0u

    fun shouldInline(): Boolean =
        false // TODO: figure out why mlir stupid
        // instrs.size < 40 // should be called on expanded blocks

    fun newVar(): IrVar =
        IrVar(Types.tbd, nextVar ++)

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
            ref(a.fn)?.let { a.fn to it }
        }
    }

    fun varUsed(variable: IrVar): Boolean =
        if (variable in rets) true
        else instrs.any { variable in it.args }

    fun inlinableCopy(nextVar: ULong, cArgs: List<IrVar>, cRets: List<IrVar>, fill: IrVar? = null): IrBlock {
        val new = IrBlock(
            name,
            ref,
            instrs.mapTo(mutableListOf()) { it.deepCopy() },
            flags.toMutableList(),
            fillArg = fillArg,
        )
        new.nextVar = max(this.nextVar, nextVar)

        val olds = new.instrs.flatMap { it.outs }.toSet()
        olds.forEach {
            val n = if (it in rets) cRets[rets.indexOf(it)]
            else new.newVar().copy(type = it.type)
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
        val newName = "${name}_\$_${fillType?.toString() ?: ""}_${inTypes.joinToString(separator = "_")}"
        if (ref(newName) != null) return newName

        val new = IrBlock(
            newName,
            ref,
            instrs.mapTo(mutableListOf()) { it.deepCopy() },
            flags.toMutableList(),
            args.toMutableList(),
            rets.toMutableList(),
        )
        new.nextVar = nextVar

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

        return newName
    }

    override fun toString(): String {
        val res = StringBuilder()

        flags.forEach {
            res.append('@')
            res.append(it)
            res.append('\n')
        }

        res.append("BLOCK ")
        res.append(name)
        fillArg?.let {
            res.append(" fill ")
            res.append(it)
        }
        res.append(" (")
        args.forEachIndexed { index, irVar ->
            if (index > 0)
                res.append(", ")
            res.append(irVar)
        }
        res.append(")\n")

        instrs.forEach {
            res.append("  ")
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

    fun type(): Type =
        Types.func(
            args.map { it.type },
            rets.map { it.type },
            fillArg?.type
        )
}