package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.*
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val boxConv = Pass<Unit>("box type conversion") { block, _ ->
    block.args.convBox()
    block.rets.convBox()
    block.fillArg?.let {
        block.fillArg = it.copy(type = it.type.convBox())
    }

    block.instrs.toList().forEach { instr ->
        instr.outs.convBox()
        instr.args.convBox()
    }
}.parallelWithoutDeepCopy()

fun Type.convBox(): Type =
    when (this) {
        is BoxType -> Types.array(of, 1).also { it.convBox() }
        is PtrType -> Types.pointer(to.convBox())
        is ArrayType -> Types.array(of.convBox(), length, vaOff)
        is FnType -> copy(
            fillType?.convBox(),
            args.map(Type::convBox),
            rets.map(Type::convBox)
        )
        else -> this
    }

fun List<IrVar>.convBoxCopy(): List<IrVar> =
    map { it.copy(type = it.type.convBox()) }

fun MutableList<IrVar>.convBox() {
    forEachIndexed { index, o ->
        this[index] = o.copy(type = o.type.convBox())
    }
}