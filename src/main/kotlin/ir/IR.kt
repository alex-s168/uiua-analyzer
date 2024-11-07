package me.alex_s168.uiua.ir

import me.alex_s168.uiua.BlockId

fun MutableList<IrVar>.updateVar(old: IrVar, new: IrVar) {
    repeat(size) { index ->
        val irVar = this[index]
        if (irVar.id == old.id)
            this[index] = new
    }
}

fun MutableMap<BlockId, IrBlock>.putBlock(block: IrBlock) {
    this[block.uid] = block
}

fun MutableMap<BlockId, IrBlock>.find(name: String) =
    this.values.find { it.name == name }