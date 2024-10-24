package me.alex_s168.uiua.ir.lower

import me.alex_s168.uiua.BoxType
import me.alex_s168.uiua.PtrType
import me.alex_s168.uiua.Type
import me.alex_s168.uiua.Types
import me.alex_s168.uiua.ir.IrVar
import me.alex_s168.uiua.ir.Pass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val boxConv = Pass<Unit>("box type conversion") { block, _ ->
    block.args.convBox()
    block.rets.convBox()

    block.instrs.toList().forEach { instr ->
        instr.outs.convBox()
        instr.args.convBox()
    }
}.parallelWithoutDeepCopy()

fun Type.convBox(): Type =
    when (this) {
        is BoxType -> Types.array(of, 1).also { it.convBox() }
        is PtrType -> Types.pointer(to.convBox())
        else -> this
    }

fun MutableList<IrVar>.convBox() {
    forEachIndexed { index, o ->
        this[index] = o.copy(type = o.type.convBox())
    }
}