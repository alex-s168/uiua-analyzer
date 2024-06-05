package me.alex_s168.uiua.ir

import me.alex_s168.uiua.Type
import me.alex_s168.uiua.Types

data class IrBlock(
    val name: String,
    val ref: (String) -> IrBlock?,
    var instrs: MutableList<IrInstr> = mutableListOf(),
    var flags: MutableList<String> = mutableListOf(),
    var args: MutableList<IrVar> = mutableListOf(),
    var rets: MutableList<IrVar> = mutableListOf(),
) {
    private var nextVar: ULong = 0u

    fun newVar(): IrVar =
        IrVar(Types.tbd, nextVar ++)

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        rets.updateVar(old, new)
        instrs.forEach {
            it.updateVar(old, new)
        }
    }

    fun instrDeclFor(variable: IrVar): IrInstr? =
        instrs.find { variable in it.outs }

    fun varUsed(variable: IrVar): Boolean =
        if (variable in rets) true
        else instrs.any { variable in it.args }

    fun expandFor(inTypes: List<Type>, putFn: (IrBlock) -> Unit): String {
        require(inTypes.size == args.size)
        val newName = "${name}_\$_${inTypes.joinToString(separator = "_")}"
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
            it.inferTypes(new, putFn)
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
}