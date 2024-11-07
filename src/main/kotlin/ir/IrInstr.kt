package me.alex_s168.uiua.ir

import blitz.collections.contents
import blitz.flatten
import me.alex_s168.uiua.*
import me.alex_s168.uiua.ast.AstNode

data class IrInstr(
    val outs: MutableList<IrVar>,
    val instr: Instr,
    var args: MutableList<IrVar>,

    val ast: AstNode? = null
) {
    fun arg(id: Int): IrVar =
        if (id >= args.size) error("instruction does not have argument ID $id: $this")
        else args[id]

    override fun equals(other: Any?): Boolean =
        other is IrInstr && instr == other.instr && args == other.args && outs == other.outs
    override fun hashCode(): Int =
        outs.hashCode() * 31 + instr.hashCode() * 31 + args.hashCode() * 31

    override fun toString(): String {
        val res = StringBuilder()

        if (outs.size > 0) {
            outs.forEachIndexed { index, irVar ->
                if (index > 0)
                    res.append(", ")
                res.append(irVar)
            }
            res.append(" = ")
        }

        val innerStr = when (instr) {
            is PrimitiveInstr -> "${Prims.all[instr.id]}${instr.param?.let { "($it)" } ?: ""}${ (if (uacPrintSpans) instr.loc?.index?.contents?.let { "@$it" } else null) ?: "" }"
            is ArrImmInstr -> "arr-make ${instr.values.flatten().contents}"
            is NumImmInstr -> "imm ${instr.value}"
            is PushFnInstr -> "fn-make ${instr.fn}"
            is CommentInstr -> "# ${instr.comment}"
            is FlagInstr -> "flag ${instr.flag}"
            is PushFnRefInstr -> "fn ${instr.fn}"
            is SourceLocInstr -> "debug ${instr.uasmSpanIdc.contents}"
            else -> instr::class.simpleName
        }
        res.append(innerStr)

        args.forEach {
            res.append(", ")
            res.append(it)
        }

        res.append(";")

        return res.toString()
    }

    fun deepCopy(): IrInstr =
        copy(
            outs = outs.toMutableList(),
            instr = instr.clone(),
            args = args.toMutableList()
        )

    fun updateVar(old: IrVar, new: IrVar) {
        args.updateVar(old, new)
        outs.updateVar(old, new)
    }

    fun similar(other: IrInstr) =
        args.map(IrVar::type).tyEq(other.args.map(IrVar::type)) &&
                outs.map(IrVar::type).tyEq(other.outs.map(IrVar::type)) &&
                instr == other.instr
}
